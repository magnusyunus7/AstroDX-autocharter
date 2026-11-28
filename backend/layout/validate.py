"""Safety net (layout step 3): the playability checker must find nothing.

The hand rules should already make every layout playable; this is the
second line. Each checker finding is traced to the moment it names and that
moment is made simpler (slide or hold -> tap, EACH -> one note, last: the
note is removed), then the chart is checked again.

Also scores how well the layout follows the song:
  on_hits       share of notes on a strong hit (slot stronger than half the song's)
  density_r     correlation of notes per bar with the planned density
"""
from __future__ import annotations

import re
from collections import defaultdict
from fractions import Fraction

import numpy as np

from backend.chartcodec.notes import ChartData, Hold, Slide, Tap
from backend.chartcodec.write import write_chart
from backend.checker.playability import check

MAX_ROUNDS = 20
_T = re.compile(r"^(-?\d+\.\d+)s:")


def body_of(notes, bpm: float) -> str:
    return write_chart(ChartData([(Fraction(0), float(bpm))], list(notes)))


def _simplify(group: list) -> list:
    """One step simpler for the notes of one moment."""
    if any(isinstance(n, (Slide, Hold)) for n in group):
        return [Tap(n.beat, n.start if isinstance(n, Slide) else n.button) if isinstance(n, (Slide, Hold)) else n
                for n in group]
    if len(group) > 1:
        return group[:1]
    return []


def make_safe(notes: list, bpm: float, first: float = 0.0) -> tuple[list, dict]:
    """Notes with no checker findings, and what had to change."""
    beat_s = 60.0 / bpm
    notes = list(notes)
    fixed, seen = 0, []
    for _ in range(MAX_ROUNDS):
        problems = check(body_of(notes, bpm), 0.0)
        if not problems:
            return notes, {"checker_fixes": fixed, "first_problems": seen[:5]}
        seen += problems[:3]
        by = defaultdict(list)
        for n in notes:
            by[n.beat].append(n)
        beats = sorted(by)
        times = np.array([float(b) * beat_s for b in beats])
        hit = set()
        for p in problems:
            m = _T.match(p)
            if not m:
                continue
            hit.add(beats[int(np.argmin(np.abs(times - float(m.group(1)))))])
        if not hit:
            break
        for b in hit:
            by[b] = _simplify(by[b])
            fixed += 1
        notes = [n for b in beats for n in by[b]]
    problems = check(body_of(notes, bpm), 0.0)
    return notes, {"checker_fixes": fixed, "first_problems": seen[:5], "left": problems}


def follow_scores(notes, plan) -> dict:
    slots = sorted({int(n.beat * 4) for n in notes})
    if not slots:
        return {"on_hits": None, "density_r": None}
    on = float(np.mean([plan.strength[s] >= 0.5 for s in slots if s < len(plan.strength)]))
    per_bar = np.bincount([s // 16 for s in slots], minlength=plan.n_bars)[:plan.n_bars].astype(float)
    m = plan.sound
    tgt = plan.bar_target[m]
    r = float(np.corrcoef(per_bar[m], tgt)[0, 1]) if per_bar[m].std() > 0 and tgt.std() > 0 else None
    return {"on_hits": round(on, 3), "density_r": None if r is None else round(r, 3)}
