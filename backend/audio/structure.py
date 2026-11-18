"""Phase 2.2: energy per bar and sections, on the confirmed bar grid.

energy    0-100 per bar: loudness + how busy the drums/bass/melody are,
          scaled within the song and lightly smoothed. Drives note density.
sections  boundaries where the music changes character (timbre, harmony,
          energy), snapped to bar lines and preferring 4/8-bar phrases.
          Each gets a similarity label (A, B, C...: same letter = sounds
          alike) and an energy level (low / mid / high / peak).
demo      a 15 s song-select preview starting at the first high-energy section.
"""
from __future__ import annotations

import numpy as np

MIN_SECTION_BARS = 4
KERNEL_BARS = 4            # novelty kernel half-width (bars)
SAME_LABEL = 0.80          # cosine similarity of section profiles to share a letter
PHRASE_SNAP = 0.2          # move a boundary 1 bar onto a 4-bar line if the change there
                           # is at least this fraction of the change at the found bar
LEVELS = ((30, "low"), (55, "mid"), (80, "high"), (101, "peak"))
DEMO_LEN = 15.0


def _bar_slices(n_bars, first, k_start, beat, sr, n_samples):
    out = []
    for b in range(n_bars):
        t0 = first + (k_start + 4 * b) * beat
        a, z = int(max(0, t0) * sr), int(max(0, t0 + 4 * beat) * sr)
        out.append((min(a, n_samples), min(z, n_samples)))
    return out


def bar_features(y, sr, first, k_start, n_beats, beat, stems: dict) -> dict:
    """Per-bar loudness (dB), activity (mean onset strength over all stems),
    and a timbre+harmony profile (MFCC + chroma means)."""
    import librosa
    n_bars = n_beats // 4
    hop = 512
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    fps = sr / hop
    loud, prof = np.full(n_bars, -80.0), np.zeros((n_bars, 25))
    for b, (a, z) in enumerate(_bar_slices(n_bars, first, k_start, beat, sr, len(y))):
        fa, fz = int(a / hop), max(int(a / hop) + 1, int(z / hop))
        if z - a < hop:
            continue
        r = rms[fa:fz]
        loud[b] = 20 * np.log10(np.sqrt(np.mean(r ** 2)) + 1e-6)
        prof[b] = np.concatenate([mfcc[:, fa:fz].mean(axis=1), chroma[:, fa:fz].mean(axis=1)])
    act = np.zeros(n_bars)
    total = sum(np.asarray(v, dtype=float) for v in stems.values())
    pts = len(total) // n_bars if n_bars else 1
    for b in range(n_bars):
        act[b] = total[b * pts:(b + 1) * pts].mean()
    return {"loud": loud, "activity": act, "profile": prof}


def _norm01(x, lo_pct=5, hi_pct=95):
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def energy_curve(feats: dict) -> np.ndarray:
    loud = feats["loud"].copy()
    silent = loud <= -79
    e = 0.55 * _norm01(loud) + 0.45 * _norm01(feats["activity"])
    e = np.convolve(np.pad(e, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    e[silent] = 0
    return np.round(e * 100).astype(int)


def _novelty(F: np.ndarray, half: int) -> np.ndarray:
    """Foote novelty: checkerboard kernel along the diagonal of the bar
    self-similarity matrix. High where 'before' and 'after' differ."""
    n = len(F)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = Fn @ Fn.T
    g = np.exp(-0.5 * (np.linspace(-1, 1, 2 * half) * 1.5) ** 2)
    G = np.outer(g, g)
    sign = np.ones((2 * half, 2 * half))
    sign[:half, half:] = sign[half:, :half] = -1
    K = G * sign
    Sp = np.pad(S, half, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        nov[i] = (Sp[i:i + 2 * half, i:i + 2 * half] * K).sum()
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-9)


def find_sections(feats: dict, energy: np.ndarray, phase: int) -> list[int]:
    """Boundary bar indices (including 0 and n). `phase` = analysis-bar index
    of bar 1, so phrase lines (bar 1, 5, 9...) are at bars b with (b - phase) % 4 == 0."""
    prof = feats["profile"]
    n = len(prof)
    if n < 2 * MIN_SECTION_BARS:
        return [0, n]
    # standardize each dimension; add energy so energy jumps count as changes
    F = np.column_stack([prof, energy[:, None] / 25.0])
    active = energy > 0
    mu, sd = F[active].mean(axis=0), F[active].std(axis=0) + 1e-6
    F = (F - mu) / sd
    nov = _novelty(F, KERNEL_BARS)
    # prefer phrase lines: 8-bar lines most, then 4-bar lines
    weight = np.array([1.35 if (b - phase) % 8 == 0 else 1.15 if (b - phase) % 4 == 0 else 1.0
                       for b in range(n)])
    score = nov * weight
    thr = max(0.25, float(np.mean(score) + 0.6 * np.std(score)))
    cands = [b for b in range(1, n) if score[b] >= thr
             and score[b] >= score[max(0, b - 1)] and score[b] >= score[min(n - 1, b + 1)]]
    # greedy by strength with a minimum section length
    chosen = []
    for b in sorted(cands, key=lambda b: -score[b]):
        if all(abs(b - c) >= MIN_SECTION_BARS for c in chosen) and b >= MIN_SECTION_BARS \
                and n - b >= MIN_SECTION_BARS // 2:
            chosen.append(b)
    # One-bar errors: a fill/riser bar before a drop already sounds different
    # (found one bar early), or a held tail hides the new section's first bar
    # (one bar late). Sections start on 4-bar lines counted from beat 1, so a
    # boundary right beside a line moves onto it if the change there is real.
    # Boundaries 2 bars from a line (genuine 6/10-bar sections) stay put.
    def snap(c):
        off = (c - phase) % 4
        if off == 1:
            line = c - 1
        elif off == 3:
            line = c + 1
        else:
            return c
        ok = 1 <= line < n and nov[line] >= PHRASE_SNAP * nov[c]
        return line if ok else c

    chosen = sorted({snap(c) for c in chosen})

    # music start/end: silent lead-in and tail become their own edges
    nz = np.nonzero(active)[0]
    if len(nz):
        for edge in (int(nz[0]), int(nz[-1]) + 1):
            if 0 < edge < n and all(abs(edge - c) >= 2 for c in chosen):
                chosen.append(edge)
    return sorted({0, n, *chosen})


def label_sections(feats: dict, energy: np.ndarray, bounds: list[int]) -> list[dict]:
    prof = feats["profile"]
    active = energy > 0
    mu, sd = prof[active].mean(axis=0), prof[active].std(axis=0) + 1e-6
    P = (prof - mu) / sd
    protos: list[tuple[str, np.ndarray]] = []
    out = []
    for a, z in zip(bounds[:-1], bounds[1:]):
        seg_e = energy[a:z]
        mean_e = float(seg_e.mean()) if len(seg_e) else 0.0
        level = next(name for lim, name in LEVELS if mean_e < lim)
        if not active[a:z].any():
            out.append({"start_bar": a, "end_bar": z, "label": "-", "level": "silent",
                        "energy": 0})
            continue
        v = P[a:z][active[a:z]].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        best, best_sim = None, -1.0
        for name, p in protos:
            sim = float(v @ p)
            if sim > best_sim:
                best, best_sim = name, sim
        if best is not None and best_sim >= SAME_LABEL:
            label = best
        else:
            label = chr(ord("A") + len(protos)) if len(protos) < 26 else "?"
            protos.append((label, v))
        out.append({"start_bar": a, "end_bar": z, "label": label, "level": level,
                    "energy": round(mean_e)})
    return out


def demo_window(sections: list[dict], first, k_start, beat, duration) -> dict:
    """Preview: the first section at the song's highest energy level present
    (peak, else high...), e.g. the first chorus rather than a loud verse."""
    if not sections:
        return {"seek": 0.0, "len": DEMO_LEN}
    for level in ("peak", "high", "mid", "low"):
        hot = [s for s in sections if s["level"] == level]
        if hot:
            pick = hot[0]
            break
    else:
        pick = sections[0]
    t = max(0.0, first + (k_start + 4 * pick["start_bar"]) * beat)
    return {"seek": round(min(t, max(0.0, duration - DEMO_LEN)), 2), "len": DEMO_LEN}


def analyze_structure(y, sr, bpm, first, k_start, n_beats, stems) -> dict:
    beat = 60.0 / bpm
    feats = bar_features(y, sr, first, k_start, n_beats, beat, stems)
    energy = energy_curve(feats)
    phase = -k_start // 4            # analysis bar index of UI bar 1
    bounds = find_sections(feats, energy, phase)
    sections = label_sections(feats, energy, bounds)
    return {"energy": energy.tolist(), "sections": sections,
            "demo": demo_window(sections, first, k_start, beat, len(y) / sr)}