"""Where in each bar the notes go: the strongest onsets on the grid.

Candidates are grid positions (16ths and triplet 8ths from level 12, 8ths and
triplets below) that something is audibly hit on. A minimum gap keeps local
speed under the level's peak (so a 314 BPM song gets 8ths, not 48 ms 16ths).
EACH goes to the strongest moments that aren't inside fast runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from backend.generator.difficulty import Targets, level_value

SUB = 12                                   # grid points per beat in the analysis
POS_HIGH = (0, 3, 4, 6, 8, 9)              # 16ths + triplet 8ths
POS_LOW = (0, 4, 6, 8)                     # 8ths + triplet 8ths
POS_BONUS = {0: 0.25, 6: 0.10, 3: 0.0, 9: 0.0, 4: -0.03, 8: -0.03}
STRAIGHT_ONLY = {3, 6, 9}                  # a beat with any of these can't take triplets
TRIPLET_ONLY = {4, 8}                      # ...and vice versa (one feel per beat)
MIN_STRENGTH = 0.15                        # weaker positions aren't candidates
GAP_FACTOR = 1.25                          # min gap = 1 / (peak_nps * this)
STREAM_SEC = 0.125                         # notes closer than this form a stream
STREAM_SLACK = 1.3                         # streams may run up to target x this


@dataclass
class Moment:
    grid: int            # index into the analysis grid (1/12 beat steps from k_start)
    strength: float
    each: bool = False


def _stem(analysis: dict, name: str) -> np.ndarray:
    return np.asarray(analysis["stems"][name], float) / 127.5      # back to 0..2


def candidate_scores(analysis: dict, targets: Targets) -> np.ndarray:
    """Score per grid point (-inf where not allowed)."""
    drums, bass, mel = (_stem(analysis, k) for k in ("drums", "bass", "melody"))
    s = np.maximum.reduce([0.9 * drums, 0.8 * bass, 1.0 * mel])
    allowed = POS_HIGH if level_value(targets.level) >= 12 else POS_LOW
    pos = np.arange(len(s)) % SUB
    bonus = np.array([POS_BONUS.get(p, 0.0) for p in range(SUB)])[pos]
    score = s + bonus * (s > MIN_STRENGTH)
    score[~np.isin(pos, allowed)] = -np.inf
    score[s < MIN_STRENGTH] = -np.inf
    return score


def min_gap_steps(analysis: dict, targets: Targets) -> int:
    step_sec = 60.0 / analysis["bpm"] / SUB
    return max(1, math.ceil((1.0 / (targets.peak_nps * GAP_FACTOR)) / step_sec - 1e-9))


def _cap_streams(moments: list[Moment], analysis: dict, targets: Targets) -> list[Moment]:
    """Break runs of fast notes longer than the level allows: drop the weakest
    note near the point where a run gets too long (a breather)."""
    limit = max(4, int(round(targets.longest_stream * STREAM_SLACK))) if targets.longest_stream else 6
    fast = max(1, int(STREAM_SEC / (60.0 / analysis["bpm"] / SUB)))
    out: list[Moment] = []
    run = 1
    for m in moments:
        if out and m.grid - out[-1].grid <= fast:
            run += 1
            if run > limit:
                # drop the weaker of this note and the previous one, restart the run
                if m.strength < out[-1].strength:
                    run = 1
                    continue
                out.pop()
                run = 1
        else:
            run = 1
        out.append(m)
    return out


def bar_capacity(analysis: dict, targets: Targets, each_frac: float) -> np.ndarray:
    """Most notes a bar can hold: its candidate positions, spaced by the min gap."""
    score = candidate_scores(analysis, targets)
    gap = min_gap_steps(analysis, targets)
    n_bars = len(score) // (4 * SUB)
    per_bar = np.isfinite(score).reshape(n_bars, 4 * SUB).sum(axis=1)
    return np.minimum(per_bar, (4 * SUB) // gap) * (1 + each_frac)


def place(analysis: dict, targets: Targets, budgets: np.ndarray) -> list[Moment]:
    score = candidate_scores(analysis, targets)
    gap = min_gap_steps(analysis, targets)
    each_frac = targets.each_pct / 100.0
    taken = np.zeros(len(score), bool)
    beat_feel: dict[int, str] = {}          # beat index -> "straight" / "triplet"
    moments: list[Moment] = []
    per_bar = 4 * SUB
    for b, notes in enumerate(budgets):
        if notes <= 0:
            continue
        want = int(round(notes / (1 + each_frac)))           # moments; EACH adds the rest
        lo, hi = b * per_bar, (b + 1) * per_bar
        order = lo + np.argsort(-score[lo:hi])
        got = 0
        for i in order:
            if got >= want or not np.isfinite(score[i]):
                break
            if taken[max(0, i - gap + 1):i + gap].any():
                continue
            p = int(i % SUB)
            feel = "straight" if p in STRAIGHT_ONLY else "triplet" if p in TRIPLET_ONLY else None
            if feel and beat_feel.setdefault(int(i // SUB), feel) != feel:
                continue                     # don't mix 16ths and triplets in one beat
            taken[i] = True
            moments.append(Moment(int(i), float(score[i])))
            got += 1
    moments.sort(key=lambda m: m.grid)
    moments = _cap_streams(moments, analysis, targets)

    # EACH: strongest moments with room around them (not inside fast runs)
    if moments:
        grids = np.array([m.grid for m in moments])
        room = np.minimum(np.diff(grids, prepend=-10 ** 6), np.diff(grids, append=10 ** 6))
        eligible = [i for i, m in enumerate(moments) if room[i] >= 2 * gap]
        n_each = int(round(each_frac * len(moments)))
        for i in sorted(eligible, key=lambda i: -moments[i].strength)[:n_each]:
            moments[i].each = True
    return moments