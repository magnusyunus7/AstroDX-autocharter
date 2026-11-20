"""Level -> generator targets.

Medians per level measured on the community chart collection (backend/
generator/priors.py, made by tools/make_priors.py from tools/corpus_stats.py),
interpolated between levels so in-between values are smooth. Touch is from DX
charts (we write DX); 15 is extrapolated (only a handful of charts).

Level values: "14" -> 14.2 and "14+" -> 14.7 (middle of each band).
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

from backend.generator.priors import LEVEL_TABLE

LEVELS = ["10", "10+", "11", "11+", "12", "12+", "13", "13+", "14", "14+", "15"]
LOG_KEYS = ("nps", "peak_nps")          # interpolated on a log scale


def level_value(level: str) -> float:
    m = re.fullmatch(r"\s*(\d+)(\+?)\s*", str(level))
    if not m:
        raise ValueError(f"bad level {level!r}; use e.g. 13, 13+, 14")
    return int(m.group(1)) + (0.7 if m.group(2) else 0.2)


def nearest_level(level: str) -> str:
    """The table level closest to `level` (for per-level distributions)."""
    L = level_value(level)
    return min(LEVELS, key=lambda lv: abs(level_value(lv) - L))


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
    bursts_pm: float = 0.0   # bursts ({24}/{32} clusters) per minute

    def to_dict(self) -> dict:
        return asdict(self)


def _value(key: str, L: float) -> float:
    pts = sorted((level_value(lv), LEVEL_TABLE[lv][key]) for lv in LEVELS)
    if L <= pts[0][0]:
        return pts[0][1]
    if L >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= L <= x1:
            f = (L - x0) / (x1 - x0)
            if key in LOG_KEYS and y0 > 0 and y1 > 0:
                return math.exp(math.log(y0) + f * (math.log(y1) - math.log(y0)))
            return y0 + f * (y1 - y0)
    return pts[-1][1]


def targets(level: str) -> Targets:
    L = level_value(level)
    return Targets(
        level=level,
        nps=round(_value("nps", L), 2),
        peak_nps=round(_value("peak_nps", L), 2),
        each_pct=round(_value("each_pct", L), 1),
        hold_pct=round(_value("hold_pct", L), 1),
        touch_pct=round(_value("touch_pct", L), 1),
        break_pct=round(_value("break_pct", L), 1),
        slides_per_min=round(_value("slides_per_min", L), 1),
        longest_stream=max(0, int(round(_value("longest_stream", L)))),
        bursts_pm=round(_value("bursts_pm", L), 2),
    )
