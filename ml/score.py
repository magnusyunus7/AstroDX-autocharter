"""Scores for generated charts, against the real chart of the same song
(Phase C3/C4). Every score is also computed for the real chart itself (or a
random version of it), so the numbers have something to be compared to.

  timing_f1    note moments that match the real chart's (within 25 ms)
  density_r    correlation of notes per bar with the real chart: does the
               chart get busier and calmer where the song does?
  kinds        share of taps / holds / slides / touch / EACH / BREAK
  flow         how much the hand flow (each move followed by the next:
               stairs, trills, spins, jumps ...) looks like real charts at
               13+: average log-likelihood per move under all training
               charts. Higher is better; random buttons score clearly lower.
  repeat       share of bars that repeat an earlier bar's pattern (in any
               rotation): real charts reuse patterns in repeated sections
  problems     playability checker findings per 100 notes (real charts get
               some too: the checker's rules are strict)
  copied       share of 48-token stretches (rotation-free) also found in
               a training chart, and the longest such run in tokens
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from backend.chartcodec.notes import Hold, Slide, Tap
from backend.chartcodec.tokens import VOCAB

SHINGLE = 48


def _moments(notes, chart):
    out = {}
    for n in notes:
        out.setdefault(n.beat, []).append(n)
    return sorted(out.items())


def note_times(notes, chart, first) -> np.ndarray:
    return np.array(sorted({first + chart.seconds(n.beat) for n in notes}))


def timing_f1(gen_t: np.ndarray, real_t: np.ndarray, tol: float = 0.025) -> float:
    if len(gen_t) == 0 or len(real_t) == 0:
        return 0.0
    j = np.clip(np.searchsorted(real_t, gen_t), 1, len(real_t) - 1)
    hit_g = np.minimum(np.abs(real_t[j] - gen_t), np.abs(real_t[j - 1] - gen_t)) <= tol
    j = np.clip(np.searchsorted(gen_t, real_t), 1, len(gen_t) - 1)
    hit_r = np.minimum(np.abs(gen_t[j] - real_t), np.abs(gen_t[j - 1] - real_t)) <= tol
    p, r = hit_g.mean(), hit_r.mean()
    return float(2 * p * r / (p + r)) if p + r else 0.0


def density_r(gen, real, n_bars: int) -> float:
    def per_bar(ns):
        c = np.zeros(n_bars)
        for n in ns:
            k = int(n.beat // 4)
            if k < n_bars:
                c[k] += 1
        return c
    a, b = per_bar(gen), per_bar(real)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def kinds(notes) -> dict:
    n = max(1, len(notes))
    c = Counter(type(x).__name__ for x in notes)
    moments = Counter(x.beat for x in notes)
    each = sum(v for v in moments.values() if v > 1)
    brk = sum(1 for x in notes if getattr(x, "brk", False))
    return {"tap": c["Tap"] / n, "hold": c["Hold"] / n, "slide": c["Slide"] / n,
            "touch": (c["Touch"] + c["TouchHold"]) / n, "each": each / n, "break": brk / n}


def _button(n):
    return n.button if isinstance(n, (Tap, Hold)) else n.start if isinstance(n, Slide) else None


def move_hist(notes) -> np.ndarray:
    """Distances (0-4 around the ring) between consecutive single-button moments."""
    seq = []
    for beat in sorted({n.beat for n in notes}):
        bs = [_button(n) for n in notes if n.beat == beat and _button(n)]
        if len(bs) == 1:
            seq.append(bs[0])
    h = np.zeros(5)
    for a, b in zip(seq, seq[1:]):
        d = abs(a - b) % 8
        h[min(d, 8 - d)] += 1
    return h / max(1, h.sum())


def flow_steps(notes) -> list[int]:
    """Moves between consecutive single-button moments, each a signed step
    around the ring (-3..4, + = clockwise)."""
    by = {}
    for n in notes:
        b = _button(n)
        if b:
            by.setdefault(n.beat, []).append(b)
    seq = [v[0] for _, v in sorted(by.items()) if len(v) == 1]
    return [((b - a + 3) % 8) - 3 for a, b in zip(seq, seq[1:])]


def flow_model(charts) -> np.ndarray:
    """How real charts follow one move with the next: log P(next move | move),
    from many charts (stairs repeat a step, trills alternate, spins keep
    going ...). 8 x 8, smoothed."""
    c = np.ones((8, 8))
    for ns in charts:
        st = flow_steps(ns)
        for x, y in zip(st, st[1:]):
            c[x + 3, y + 3] += 1
    return np.log(c / c.sum(1, keepdims=True))


def flow_ll(notes, model: np.ndarray) -> float | None:
    """Average log-likelihood per move under real charts' flow (higher = moves
    more like real charts; random buttons score clearly lower)."""
    st = flow_steps(notes)
    if len(st) < 2:
        return None
    return float(np.mean([model[x + 3, y + 3] for x, y in zip(st, st[1:])]))


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def randomized(notes, seed=0):
    """The same rhythm with random buttons: the 'random chart' baseline."""
    from dataclasses import replace
    rng = np.random.default_rng(seed)
    out = []
    for n in notes:
        if isinstance(n, (Tap, Hold)):
            out.append(replace(n, button=int(rng.integers(1, 9))))
        elif isinstance(n, Slide):
            out.append(replace(n, start=int(rng.integers(1, 9))))
        else:
            out.append(n)
    return out


def repeat_share(notes) -> float:
    """Bars (with 2+ notes) whose pattern, rotation-free, already appeared earlier."""
    bars = {}
    for n in notes:
        bars.setdefault(int(n.beat // 4), []).append(n)
    seen, rep, tot = set(), 0, 0
    for k in sorted(bars):
        ns = sorted(bars[k], key=lambda x: (x.beat, str(_button(x))))
        if len(ns) < 2:
            continue
        b0 = next((_button(x) for x in ns if _button(x)), 0)
        key = tuple((str(x.beat - 4 * k), type(x).__name__, ((_button(x) or 0) - b0) % 8 if _button(x) else
                     getattr(x, "sensor", "")) for x in ns)
        tot += 1
        rep += key in seen
        seen.add(key)
    return rep / max(1, tot)


def problems_per_100(notes, chart, first) -> float | None:
    """Playability checker findings per 100 notes (single-BPM charts only)."""
    if len(chart.tempo) > 1 or not notes:
        return None
    from backend.checker.playability import check
    from ml.generate import chart_text
    body = chart_text(notes, [(0.0, chart.tempo[0][1])])
    return 100.0 * len(check(body, first)) / len(notes)


def rotation_free(tokens) -> list[str]:
    """Tokens with each button written relative to the previous one (so a
    rotated copy of a chart looks the same)."""
    out, prev = [], None
    for t in tokens:
        s = VOCAB[t]
        if s[0] == "B" and s[1:].isdigit():
            b = int(s[1:])
            out.append(f"d{(b - prev) % 8 if prev else 0}")
            prev = b
        else:
            out.append(s)
    return out


def _h(parts) -> int:
    """Stable 64-bit hash (Python's own hash() changes between runs)."""
    return int.from_bytes(hashlib.blake2b(" ".join(parts).encode(), digest_size=8).digest(), "little", signed=True)


def shingles(tokens, n: int = SHINGLE) -> set[int]:
    r = rotation_free(tokens)
    return {_h(r[i:i + n]) for i in range(0, max(0, len(r) - n + 1))}


def copied(tokens, train_shingles: set[int], n: int = SHINGLE) -> tuple[float, int]:
    r = rotation_free(tokens)
    hits = [_h(r[i:i + n]) in train_shingles for i in range(0, max(0, len(r) - n + 1))]
    if not hits:
        return 0.0, 0
    longest = run = 0
    for h in hits:
        run = run + 1 if h else 0
        longest = max(longest, run)
    return float(np.mean(hits)), (longest + n - 1) if longest else 0
