"""Build a full maidata.txt.

Difficulty slots (simai / AstroDX convention):
    1 Easy, 2 Basic, 3 Advanced, 4 Expert, 5 Master, 6 Re:Master, 7 Original/Utage
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from backend.models.chart import BEATS_PER_BAR, Chart
from backend.simai.serializer import fmt_number, serialize

SLOT_NAMES = {1: "Easy", 2: "Basic", 3: "Advanced", 4: "Expert",
              5: "Master", 6: "Re:Master", 7: "Original"}


@dataclass
class DifficultyChart:
    slot: int               # 1..7
    level: str              # "14+", "13", ...
    chart: Chart
    designer: str = ""


@dataclass
class MaidataMeta:
    title: str
    artist: str = ""
    first: float = 0.0      # seconds from audio start to bar 1 beat 1
    designer: str = "chartgen"
    extra: dict[str, str] = field(default_factory=dict)  # any other &key=value


def _clean(v: str) -> str:
    # '&' and newlines would break the key=value format
    return str(v).replace("\r", " ").replace("\n", " ").replace("&", "＆")


def rebase_first(first: float, bpm: float) -> tuple[float, int]:
    """Move &first back by whole bars so it sits near the start of the audio.

    Returns (new_first, bars_added). Shift every note forward by
    bars_added * 4 beats so their audio time is unchanged. Keeps MajdataEdit's
    view lined up with the song from 0:00 instead of starting mid-song.
    """
    bar_sec = 60.0 / bpm * BEATS_PER_BAR
    if first < 0:
        raise ValueError("first must be >= 0 (put the first downbeat inside the audio)")
    bars = int(math.floor(first / bar_sec + 1e-9))
    return round(first - bars * bar_sec, 6), bars


def build_maidata(meta: MaidataMeta, charts: list[DifficultyChart],
                  audio_duration: float | None = None) -> str:
    if not charts:
        raise ValueError("need at least one chart")
    slots = [c.slot for c in charts]
    if len(set(slots)) != len(slots):
        raise ValueError(f"duplicate difficulty slots: {slots}")

    lines = [
        f"&title={_clean(meta.title)}",
        f"&artist={_clean(meta.artist)}",
        f"&wholebpm={fmt_number(charts[0].chart.bpm)}",
        f"&first={fmt_number(meta.first)}",
        f"&des={_clean(meta.designer)}",
    ]
    for k, v in meta.extra.items():
        lines.append(f"&{k}={_clean(v)}")

    for c in sorted(charts, key=lambda c: c.slot):
        if c.slot not in SLOT_NAMES:
            raise ValueError(f"bad slot {c.slot}")
        if audio_duration is not None:
            end_sec = meta.first + float(c.chart.last_beat) * 60.0 / c.chart.bpm
            if end_sec > audio_duration:
                raise ValueError(
                    f"{SLOT_NAMES[c.slot]} chart ends at {end_sec:.2f}s but audio is "
                    f"{audio_duration:.2f}s long")
        lines.append(f"&lv_{c.slot}={_clean(c.level)}")
        lines.append(f"&des_{c.slot}={_clean(c.designer or meta.designer)}")
        lines.append(f"&inote_{c.slot}={serialize(c.chart)}")

    return "\n".join(lines) + "\n"