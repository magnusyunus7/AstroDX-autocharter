"""Layout generator: a playable, simple chart draft that follows the song,
for chart makers to finish ("layout" in Geometry Dash terms; decoration and
polish are left to the creator).

    from backend.layout import make_layout
    lay = make_layout(mel_db, bpm, first, duration, density="medium")
    lay.body          # simai note text for &inote_N=
    lay.stats         # counts and the follow-the-song scores

  features.py   audio measurements (shared with tools/fit_layout_tables.py)
  rhythm.py     when: a note budget per bar (calm parts get fewer), then the
                best 16th slots; which moments are EACH / holds / slides
  hands.py      the two-hand rules (sides, reach, alternation, busy hands)
  place.py      where: buttons by the flow real charts use, repeated phrases
                mirrored
  validate.py   the checker must find nothing; follow-the-song scores
  tables.json   what was learned from real charts (tools/fit_layout_tables.py)

Densities: basic, light, medium, dense (difficulty groups below 7, 7-9,
10-12, 13 and harder).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

TABLES = Path(__file__).with_name("tables.json")
DENSITIES = ("basic", "light", "medium", "dense")


@lru_cache(maxsize=4)
def load_tables(path: str | None = None) -> dict:
    return json.loads(Path(path or TABLES).read_text(encoding="utf-8"))


@dataclass
class Layout:
    notes: list
    body: str
    level: str
    stats: dict = field(default_factory=dict)


def make_layout(mel_db, bpm: float, first: float, duration: float, density: str = "medium",
                seed: int = 0, scale: float = 1.0, tables: dict | None = None) -> Layout:
    """mel_db: (frames, 80) log-mel (features.mel). first: seconds of the
    first downbeat (beat 0). scale: more (>1) or fewer (<1) notes overall."""
    from backend.layout import place, rhythm, validate
    tables = tables or load_tables()
    pl = rhythm.plan(mel_db, bpm, first, duration, density, tables, scale)
    notes, pinfo = place.place(pl.moments, tables["flow"][density], bpm, first, seed)
    notes, vinfo = validate.make_safe(notes, bpm)
    stats = {**pl.info, **pinfo, **vinfo, **validate.follow_scores(notes, pl), "notes": len(notes)}
    return Layout(notes, validate.body_of(notes, bpm), rhythm.LEVEL_OF[density], stats)


def maidata_text(title: str, artist: str, bpm: float, first: float, charts: dict[int, tuple[str, list]]) -> str:
    """maidata.txt for one or more layouts. charts: {slot: (level, notes)},
    notes with beat 0 at `first` seconds. &first is moved back by whole bars
    to near the start of the audio (notes shifted to match), so editors show
    the song from 0:00."""
    from backend.layout.validate import body_of
    from dataclasses import replace
    bar = 4 * 60.0 / bpm
    bars = int(first // bar) if first > 0 else 0
    shift = Fraction(4 * bars)
    lines = [f"&title={title}", f"&artist={artist}", f"&wholebpm={bpm:g}",
             f"&first={first - bars * bar:.4f}", "&des=AstroDX autocharter (layout)"]
    for slot, (level, notes) in sorted(charts.items()):
        moved = [replace(n, beat=n.beat + shift) for n in notes]
        lines += [f"&lv_{slot}={level}", f"&inote_{slot}={body_of(moved, bpm)}"]
    return "\n".join(lines) + "\n"


__all__ = ["make_layout", "maidata_text", "load_tables", "Layout", "DENSITIES"]
