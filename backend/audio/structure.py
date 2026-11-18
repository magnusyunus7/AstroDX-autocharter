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
KERNEL_BARS = 8            # novelty kernel half-width (units): wide enough that small
                           # 4-bar variations inside a section average out
REFINE_KERNEL = 4          # bar-level kernel used to place each boundary on the exact bar
SAME_LABEL = 0.80          # cosine similarity of section profiles to share a letter
# Choosing the boundary set (dynamic programming over candidate units):
BOUNDARY_COST = 0.4        # each boundary must "pay" for itself with novelty (0..1)
PEN_NOT_4 = 0.3            # extra cost: section length not a multiple of 4 units
PEN_NOT_8 = 0.1            # extra cost: multiple of 4 but not of 8. First/last section exempt.
                           # (Penalties, not bonuses: a bonus can make splitting free.)
CAND_MIN = 0.15            # novelty peaks considered (plus their +-1 unit neighbours)
FAST_BPM = 230             # at/above this, sections are found in 2-bar units (half-time):
                           # a 314 BPM bar is only 0.76 s
LABEL_SIM = 0.55           # sections at least this similar (cosine) share a letter
MAX_LABELS = 8             # keep merging the most similar groups beyond this
FILL_DROP_DB = 6.0         # a bar whose sub-120 Hz energy (kick + bass fundamentals) is this
                           # much below the bars on both sides is a fill/riser bar
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
    # sub-120 Hz energy per bar (dB): kick and bass fundamentals. Snare rolls
    # and risers have little energy down here, so fills show up as dips.
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    sub = S[(freqs >= 30) & (freqs < 120)].sum(axis=0)
    low_end = np.full(n_bars, -100.0)
    for b, (a, z) in enumerate(_bar_slices(n_bars, first, k_start, beat, sr, len(y))):
        fa, fz = a // 1024, max(a // 1024 + 1, z // 1024)
        if z - a >= 1024:
            low_end[b] = 10 * np.log10(sub[fa:fz].mean() + 1e-10)
    return {"loud": loud, "activity": act, "profile": prof, "low_end": low_end}


def _norm01(x, lo_pct=5, hi_pct=95):
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def energy_curve(feats: dict, smooth: bool = True) -> np.ndarray:
    """0-100 per bar. Smoothed for display/density; unsmoothed for section
    detection (smoothing leaks a section change one bar backwards)."""
    loud = feats["loud"].copy()
    silent = loud <= -79
    e = 0.55 * _norm01(loud) + 0.45 * _norm01(feats["activity"])
    if smooth:
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
    # scale by a high percentile, not the max: one huge change (e.g. a
    # breakdown) shouldn't shrink every other boundary below the cost
    ref = np.percentile(nov, 95) if len(nov) > 20 else nov.max()
    return np.clip(nov / (ref + 1e-9), 0, 1.5)


def find_sections(feats: dict, energy: np.ndarray, phase: int = 0) -> list[int]:
    """Boundary bar indices (including 0 and n).

    Picks the whole set of boundaries at once: strong changes (novelty), plus
    a bonus when consecutive boundaries are a multiple of 4/8 bars apart
    (sections are 4/8/16 bars long, wherever the phrases start), minus a cost
    per boundary. This fixes one-bar errors (a fill/riser bar before a drop
    sounds different already; a held tail hides a section's first bar) without
    assuming phrases start at bar 1. `phase` is unused (kept for callers)."""
    prof = feats["profile"]
    n = len(prof)
    if n < 2 * MIN_SECTION_BARS:
        return [0, n]
    F = np.column_stack([prof, energy[:, None] / 25.0])
    active = energy > 0
    mu, sd = F[active].mean(axis=0), F[active].std(axis=0) + 1e-6
    F = (F - mu) / sd
    nov = _novelty(F, KERNEL_BARS)

    # Fill/riser bars: the kick and bass usually drop out for the transition
    # bar and come back on the drop. A section never starts on such a bar (it
    # ends the previous section), so it's replaced by the bar after it before
    # the boundary set is chosen.
    low = feats.get("low_end")

    def is_fill(c):
        return low is not None and _is_fill(low, c, active)

    peaks = [b for b in range(1, n) if nov[b] >= CAND_MIN
             and nov[b] >= nov[b - 1] and nov[b] >= nov[min(n - 1, b + 1)]]
    cands = set()
    for b in peaks:
        for c in (b - 1, b, b + 1):
            if 1 <= c < n:
                cands.add(_past_fills(low, c, n, active=active) if low is not None else c)
    # a boundary moved off a fill keeps the fill's (stronger) novelty
    nov = nov.copy()
    for c in range(1, n - 1):
        if is_fill(c):
            d = _past_fills(low, c, n, active=active)
            nov[d] = max(nov[d], nov[c])
    cands = sorted(cands)
    pts = [0] + cands + [n]

    def length_pen(length, first_or_last):
        if first_or_last:
            return 0.0
        return 0.0 if length % 8 == 0 else PEN_NOT_8 if length % 4 == 0 else PEN_NOT_4

    # best[j] = best score of a segmentation of [0, pts[j]] ending with a boundary at pts[j]
    best = [-np.inf] * len(pts)
    prev = [-1] * len(pts)
    best[0] = 0.0
    for j in range(1, len(pts)):
        cj = pts[j]
        gain = 0.0 if cj == n else nov[cj] - BOUNDARY_COST
        for i in range(j):
            ci = pts[i]
            if best[i] == -np.inf:
                continue
            length = cj - ci
            if length < MIN_SECTION_BARS and not (cj == n and length >= MIN_SECTION_BARS // 2):
                continue
            sc = best[i] + gain - length_pen(length, ci == 0 or cj == n)
            if sc > best[j]:
                best[j], prev[j] = sc, i
    chosen, j = [], len(pts) - 1
    while j > 0:
        chosen.append(pts[j])
        j = prev[j]
        if j == -1:
            break
    chosen = [c for c in chosen if c != n]

    # music start/end: silent lead-in and tail become their own edges
    nz = np.nonzero(active)[0]
    if len(nz):
        for edge in (int(nz[0]), int(nz[-1]) + 1):
            if 0 < edge < n and all(abs(edge - c) >= 2 for c in chosen):
                chosen.append(edge)
    return sorted({0, n, *chosen})


def label_sections(feats: dict, energy: np.ndarray, bounds: list[int]) -> list[dict]:
    """Letters by similarity, grouped over the whole song: average-linkage
    merging of section profiles while groups are at least LABEL_SIM similar,
    then keep merging the closest groups until there are <= MAX_LABELS.
    Letters are assigned in order of first appearance."""
    prof = feats["profile"]
    active = energy > 0
    mu, sd = prof[active].mean(axis=0), prof[active].std(axis=0) + 1e-6
    P = (prof - mu) / sd
    segs, vecs = [], []
    for a, z in zip(bounds[:-1], bounds[1:]):
        seg_e = energy[a:z]
        mean_e = float(seg_e.mean()) if len(seg_e) else 0.0
        level = next(name for lim, name in LEVELS if mean_e < lim)
        silent = not active[a:z].any()
        segs.append({"start_bar": a, "end_bar": z, "label": "-", "level": "silent" if silent else level,
                     "energy": 0 if silent else round(mean_e)})
        if silent:
            vecs.append(None)
        else:
            v = P[a:z][active[a:z]].mean(axis=0)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
    idx = [i for i, v in enumerate(vecs) if v is not None]
    groups = [[i] for i in idx]

    def gsim(g1, g2):
        return float(np.mean([vecs[i] @ vecs[j] for i in g1 for j in g2]))

    while len(groups) > 1:
        best, pair = -2.0, None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                sim = gsim(groups[x], groups[y])
                if sim > best:
                    best, pair = sim, (x, y)
        if best < LABEL_SIM and len(groups) <= MAX_LABELS:
            break
        x, y = pair
        groups[x] += groups.pop(y)
    order = sorted(groups, key=min)
    for k, g in enumerate(order):
        for i in g:
            segs[i]["label"] = chr(ord("A") + k)
    return segs


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


def _bar_novelty(feats, energy):
    F = np.column_stack([feats["profile"], energy[:, None] / 25.0])
    active = energy > 0
    if active.sum() < 2:
        return np.zeros(len(energy))
    F = (F - F[active].mean(axis=0)) / (F[active].std(axis=0) + 1e-6)
    return _novelty(F, REFINE_KERNEL)


def _is_fill(low, c, active=None):
    """Transition bar: far less sub-bass than the bars after it (kick and bass
    come back on the drop) and not more than the bars before it (so also a
    fill coming out of a bass-less breakdown)."""
    before, after = low[max(0, c - 3):c], low[c + 1:c + 4]
    if len(before) == 0 or len(after) == 0:
        return False
    # a fill sits *between* sections: the bar before it must have music
    # (the first bar after silence is just the song starting)
    if active is not None and not active[c - 1]:
        return False
    gap = np.median(after) - low[c]
    # far below what follows (a riser out of a breakdown can still have some
    # low end, but nowhere near the drop), or below what follows and not
    # above what came before (kick/bass drop out for the fill)
    return gap > 2 * FILL_DROP_DB or (gap > FILL_DROP_DB and low[c] <= np.median(before) + 3.0)


def _past_fills(low, c, n, max_steps=2, active=None):
    """Move forward past up to `max_steps` consecutive fill bars."""
    for _ in range(max_steps):
        if c + 1 < n and _is_fill(low, c, active):
            c += 1
        else:
            break
    return c


def _refine_to_bars(feats, energy, bars: list[int], unit: int) -> list[int]:
    """Place each boundary (found on a coarser scale) on the exact bar: the bar
    within +-unit with the strongest bar-level change, never on a fill bar."""
    n = len(energy)
    nov = _bar_novelty(feats, energy)
    low = feats["low_end"]
    out = {0, n}
    for b in bars:
        if b <= 0 or b >= n:
            continue
        options = [c for c in range(b - unit, b + unit + 1) if 1 <= c < n]
        options = [_past_fills(low, c, n, active=energy > 0) for c in options]
        out.add(max(options, key=lambda c: nov[c]))
    return sorted(out)


def _to_units(feats: dict, energy: np.ndarray, unit: int):
    """Average consecutive bars into units of `unit` bars (half-time view of
    very fast songs). Units start at analysis bar 0."""
    if unit == 1:
        return feats, energy
    n = len(energy)
    m = (n + unit - 1) // unit
    def agg(x, fn=np.mean):
        return np.array([fn(x[i * unit:(i + 1) * unit], axis=0) for i in range(m)])
    out = {"loud": agg(feats["loud"]), "activity": agg(feats["activity"]),
           "profile": agg(feats["profile"]), "low_end": agg(feats["low_end"])}
    return out, np.round(agg(energy.astype(float))).astype(int)


def analyze_structure(y, sr, bpm, first, k_start, n_beats, stems) -> dict:
    beat = 60.0 / bpm
    feats = bar_features(y, sr, first, k_start, n_beats, beat, stems)
    energy = energy_curve(feats)
    phase = -k_start // 4            # analysis bar index of UI bar 1
    unit = 2 if bpm >= FAST_BPM else 1
    ufeats, uenergy = _to_units(feats, energy_curve(feats, smooth=False), unit)
    ubounds = find_sections(ufeats, uenergy, phase // unit)
    n = len(energy)
    bounds = _refine_to_bars(feats, energy_curve(feats, smooth=False),
                             [b * unit for b in ubounds], unit)
    sections = label_sections(feats, energy, bounds)
    return {"energy": energy.tolist(), "sections": sections,
            "demo": demo_window(sections, first, k_start, beat, len(y) / sr)}