"""How many notes each bar gets.

The level gives the average notes/second and the busiest-4-seconds cap
(difficulty.py). The song decides where they go: each bar's weight comes from
its energy and its section's level (the user's ✎ override wins). Quiet bars
(breakdowns) get few notes, bars after a tempo drift get none.
"""
from __future__ import annotations

import numpy as np

from backend.generator.difficulty import Targets

LEVEL_WEIGHT = {"low": 0.45, "mid": 0.75, "high": 1.0, "peak": 1.25, "silent": 0.0}
QUIET_WEIGHT = 0.5         # breakdown / quiet bars (still a few notes)
PEAK_HEADROOM = 1.0        # a single bar may reach peak_nps x this
CONTRASTS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)   # weight exponents tried to reach the peak target


def section_level_per_bar(analysis: dict, overrides: dict[str, str]) -> list[str]:
    n = len(analysis["energy"])
    out = ["silent"] * n
    for s in analysis["sections"]:
        lvl = overrides.get(str(s["start_bar"]), s["level"])
        for b in range(s["start_bar"], min(n, s["end_bar"])):
            out[b] = lvl
    return out


def playable_range(analysis: dict) -> tuple[int, int]:
    """[first, last) bars to chart: from the first bar with energy to the last."""
    e = np.asarray(analysis["energy"])
    nz = np.nonzero(e > 0)[0]
    if not len(nz):
        return 0, 0
    return int(nz[0]), int(nz[-1]) + 1


def drifting_bars(analysis: dict, run: int = 3) -> np.ndarray:
    """Bars inside runs of `run`+ 'off' bars (quiet bars don't break a run):
    the grid doesn't fit there, so no notes. Elsewhere the song is charted,
    e.g. a mismatched intro doesn't stop the rest of the chart."""
    st = [b["status"] for b in analysis["bars"]]
    mask = np.zeros(len(st), bool)
    i = 0
    while i < len(st):
        if st[i] != "off":
            i += 1
            continue
        j, offs = i, 0
        while j < len(st) and st[j] in ("off", "quiet"):
            offs += st[j] == "off"
            j += 1
        if offs >= run:
            mask[i:j] = True
        i = j
    return mask


def bar_weights(analysis: dict, overrides: dict[str, str]) -> np.ndarray:
    e = np.asarray(analysis["energy"], float) / 100.0
    levels = section_level_per_bar(analysis, overrides)
    status = [b["status"] for b in analysis["bars"]]
    w = (0.25 + 0.75 * e ** 1.5) * np.array([LEVEL_WEIGHT[l] for l in levels])
    w *= np.array([QUIET_WEIGHT if s == "quiet" else 1.0 for s in status])
    w[e <= 0] = 0
    w[drifting_bars(analysis)] = 0
    a, z = playable_range(analysis)
    w[:a] = 0
    w[z:] = 0
    return w


def note_budgets(analysis: dict, targets: Targets, overrides: dict[str, str],
                 capacity: np.ndarray | None = None) -> np.ndarray:
    """Integer notes per bar. The average matches targets.nps; the contrast
    between calm and intense bars is sharpened until the busiest 4 seconds
    match targets.peak_nps (real charts concentrate notes in the intense
    parts instead of spreading them evenly)."""
    beat = 60.0 / analysis["bpm"]
    bar_sec = 4 * beat
    base = bar_weights(analysis, overrides)
    win = max(1, int(round(4.0 / bar_sec)))           # bars in a 4 s window
    best, best_err = None, None
    for gamma in CONTRASTS:
        w = base ** gamma
        bud = _fill(w, analysis, targets, capacity, bar_sec)
        if bud.sum() <= 0:
            return bud
        peak = np.convolve(bud, np.ones(win), mode="valid").max() / (win * bar_sec)
        err = abs(peak - targets.peak_nps)
        if best is None or err < best_err:
            best, best_err = bud, err
        if peak >= targets.peak_nps:
            break
    return best


def _fill(w, analysis, targets, capacity, bar_sec) -> np.ndarray:
    a, z = playable_range(analysis)
    if z <= a or w.sum() <= 0:
        return np.zeros(len(w), int)
    total = targets.nps * (z - a) * bar_sec
    cap = np.full(len(w), targets.peak_nps * PEAK_HEADROOM * bar_sec)
    if capacity is not None:
        cap = np.minimum(cap, capacity)
    budget = np.zeros(len(w))
    free = w > 0
    remaining = total
    for _ in range(8):                       # water-filling against the caps
        if remaining <= 1e-6 or not free.any():
            break
        share = remaining * w * free / (w * free).sum()
        budget += share
        over = budget > cap
        remaining = float((budget - cap)[over].sum())
        budget[over] = cap[over]
        free &= ~over
    # integer notes, carrying the rounding error forward
    out, carry = np.zeros(len(w), int), 0.0
    for i, v in enumerate(budget):
        x = v + carry
        out[i] = int(np.floor(x + 0.5))
        carry = x - out[i]
    return out