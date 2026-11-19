"""Level -> generator targets.

Fitted on 41 real charts (levels 3-14+, tools/chart_stats.py --fit) with
outliers trimmed: slow, reading-heavy charts like a sparse 14 at 85 BPM are
real but not what "typical density" should aim for. Smooth in level, so
levels without sample charts are still sensible and nothing goes backwards.

Level values: "14" -> 14.2 and "14+" -> 14.7 (middle of each band).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

LEVELS = ["10", "10+", "11", "11+", "12", "12+", "13", "13+", "14", "14+", "15"]

# (a, b, log): value = exp(a + b*L) if log else a + b*L.  Refit with
# `python -m tools.chart_stats samples/ --fit` when you add more charts.
FIT = {
    "nps":            (-1.5564, 0.2421, True),
    "peak_nps":       (-0.7519, 0.2394, True),
    "slide_pm":       (-32.2591, 4.7295, False),
    "each_pct":       (78.6593, -3.6598, False),
    "hold_pct":       (35.2282, -1.8911, False),
    "touch_pct":      (11.3122, -0.4186, False),
    "break_pct":      (17.5859, -0.7912, False),
    "longest_stream": (-197.7103, 16.4828, False),
}


def level_value(level: str) -> float:
    m = re.fullmatch(r"\s*(\d+)(\+?)\s*", str(level))
    if not m:
        raise ValueError(f"bad level {level!r}; use e.g. 13, 13+, 14")
    return int(m.group(1)) + (0.7 if m.group(2) else 0.2)


@dataclass
class Targets:
    level: str
    nps: float               # average notes per second over the chart
    peak_nps: float          # notes per second in the busiest 4 s
    each_pct: float          # % of note moments that are EACH (2+ notes at once)
    hold_pct: float          # % of notes that are holds
    touch_pct: float         # % of notes that are touch notes
    break_pct: float         # % of notes marked BREAK
    slides_per_min: float
    longest_stream: int      # longest run of fast consecutive notes (0 = none)

    def to_dict(self) -> dict:
        return asdict(self)


def _f(key: str, L: float) -> float:
    a, b, log = FIT[key]
    return math.exp(a + b * L) if log else a + b * L


def targets(level: str) -> Targets:
    L = level_value(level)
    return Targets(
        level=level,
        nps=round(_f("nps", L), 2),
        peak_nps=round(_f("peak_nps", L), 2),
        each_pct=round(max(10.0, _f("each_pct", L)), 1),
        hold_pct=round(max(3.0, _f("hold_pct", L)), 1),
        touch_pct=round(max(0.0, _f("touch_pct", L)), 1),
        break_pct=round(max(2.0, _f("break_pct", L)), 1),
        slides_per_min=round(max(0.0, _f("slide_pm", L)), 1),
        longest_stream=max(0, int(round(_f("longest_stream", L)))),
    )