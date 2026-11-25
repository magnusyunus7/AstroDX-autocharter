"""Chart statistics: notes per second, busiest 4 s, note-type mix, EACH
share, longest stream, slides per minute: what a chart contains."""
from __future__ import annotations

import re

import numpy as np

from backend.simai.reader import note_groups

SLIDE_CHARS = set("-<>^vpqszwV")


def classify(tok: str) -> list[str]:
    """Note kinds in one token (a slide token with '*' has several paths)."""
    t = re.sub(r"\[[^\]]*\]", "", tok)          # drop durations
    kinds = []
    brk = "b" in t
    if t[:1] in "ABCDE":
        kinds.append("touch_hold" if "h" in t else "touch")
    elif any(c in SLIDE_CHARS for c in t[1:]):
        kinds.append("slide_star")              # the star you tap
        kinds += ["slide"] * (t.count("*") + 1)
    elif "h" in t:
        kinds.append("hold")
    else:
        kinds.append("tap")
    if brk:
        kinds.append("break")
    if "x" in t:
        kinds.append("ex")
    return kinds


def stats(body: str, first: float) -> dict:
    groups = note_groups(body, first)
    if not groups:
        return {}
    times = np.array([t for t, _ in groups])
    sizes = np.array([len(n) for _, n in groups])
    kinds: dict[str, int] = {}
    for _, notes in groups:
        for tok in notes:
            for k in classify(tok):
                kinds[k] = kinds.get(k, 0) + 1
    hits = sizes.sum()                           # notes you hit (slide paths not counted twice)
    span = max(1.0, times[-1] - times[0])
    # busiest 4 s window
    peak = 0
    j = 0
    for i in range(len(times)):
        while times[i] - times[j] > 4.0:
            j += 1
        peak = max(peak, sizes[j:i + 1].sum())
    # longest stream: consecutive groups <= 1/8 s apart (fast runs)
    gaps = np.diff(times)
    run = best = 1
    for g in gaps:
        run = run + 1 if g <= 0.125 else 1
        best = max(best, run)
    return {
        "notes": int(hits),
        "nps": hits / span,
        "peak_nps": peak / 4.0,
        "each_pct": 100 * (sizes > 1).sum() / len(sizes),
        "tap_pct": 100 * kinds.get("tap", 0) / hits,
        "hold_pct": 100 * kinds.get("hold", 0) / hits,
        "slide_pm": kinds.get("slide", 0) / span * 60,
        "touch_pct": 100 * (kinds.get("touch", 0) + kinds.get("touch_hold", 0)) / hits,
        "break_pct": 100 * kinds.get("break", 0) / hits,
        "longest_stream": int(best),
        "length_s": span,
    }