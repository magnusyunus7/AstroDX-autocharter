"""Charts of one song for the dataset (Phase C2): parse, label, check.

Every chart in maidata.txt (slots 1-6) becomes a record with its labels and
the facts needed to decide whether it's used for training. Nothing is
thrown away here: excluded charts keep their reason, so the choice can be
changed later without rebuilding.

Labels
  level   &lv_N as written ("13+") and as a number (13+ -> 13.7, "?" -> None)
  era     STD for versions 01-13 (up to FiNALE), DX from 14 (DX) on
  source  "converted" (straight ma2 -> maidata: versions up to UNiVERSE PLUS,
          and Basic/Advanced after that) or "proofread" (Expert, Master and
          Re:Master from FESTiVAL on, converted by hand and checked)
"""
from __future__ import annotations

import hashlib
import re
from fractions import Fraction

from backend.chartcodec.notes import ChartData
from backend.chartcodec.parse import ParseError, parse_chart, parse_maidata
from backend.chartcodec.tokens import ids, encode

SLOTS = range(1, 7)
STD_UP_TO = 13                 # 01-13 STD, 14+ DX
CONVERTED_UP_TO = 19           # up to UNiVERSE PLUS everything is a straight conversion
PROOFREAD_SLOTS = (4, 5, 6)    # from FESTiVAL on: Expert, Master, Re:Master by hand
MIN_FIT = 0.90                 # charts whose beats follow the music less than this are left out
FIT_DENOMS = (48, 5, 7, 9, 10)


def version_number(version: str) -> int | None:
    m = re.match(r"(\d+)", version)
    return int(m.group(1)) if m else None


def era_of(version: str) -> str:
    v = version_number(version)
    return "STD" if v is not None and v <= STD_UP_TO else "DX"


def source_of(version: str, slot: int) -> str:
    v = version_number(version)
    if v is not None and v > CONVERTED_UP_TO and slot in PROOFREAD_SLOTS:
        return "proofread"
    return "converted"


def level_value(text: str) -> float | None:
    """'13' -> 13.0, '13+' -> 13.7, '13.4' -> 13.4, anything else -> None."""
    m = re.fullmatch(r"\s*(\d+)(\+?)(?:\.(\d))?\s*", text or "")
    if not m:
        return None
    if m.group(3):
        return float(f"{m.group(1)}.{m.group(3)}")
    return int(m.group(1)) + (0.7 if m.group(2) else 0.0)


def title_key(title: str) -> str:
    """The same song in different folders -> the same key (case, spaces, symbols ignored)."""
    return "".join(ch for ch in title.casefold() if ch.isalnum())


def grid_fit(chart: ChartData) -> float:
    """Share of note moments within 3 ms of a 1/48-beat or 5th/7th/9th/10th
    position: how well the chart's beats follow the music (see C1)."""
    beats = sorted({n.beat for n in chart.notes})
    if not beats:
        return 0.0
    ok = 0
    for b in beats:
        ms = min(abs(float(b - Fraction(round(b * d), d))) for d in FIT_DENOMS) * 60000 / chart.bpm_at(b)
        ok += ms <= 3
    return ok / len(beats)


def chart_hash(chart: ChartData) -> str:
    """Same notes and tempo -> same hash (to find one chart copied into two versions)."""
    h = hashlib.sha1()
    h.update(repr(chart.tempo).encode())
    for n in chart.notes:
        h.update(repr(n).encode())
    return h.hexdigest()[:16]


def read_song(text: str, version: str) -> tuple[dict, list[dict]]:
    """maidata.txt text -> (song info, chart records). Each record has
    'use' False with a 'reason' when the chart can't be used."""
    meta = parse_maidata(text)
    song = {"title": meta["title"], "first": meta["first"], "wholebpm": meta["wholebpm"]}
    records = []
    for slot, c in sorted(meta["charts"].items()):
        if slot not in SLOTS:
            continue
        rec = {"slot": slot, "level": c["level"], "level_value": level_value(c["level"]),
               "era": era_of(version), "source": source_of(version, slot), "use": True, "reason": ""}
        try:
            chart, issues = parse_chart(c["body"], meta["wholebpm"], repair=True)
        except ParseError as e:
            rec.update(use=False, reason=f"parse: {e}")
            records.append(rec)
            continue
        if not chart.notes:
            continue                                         # empty / placeholder "(280)E"
        view, moved = chart.model_view()
        rec.update(
            chart=chart, view=view, tokens=ids(encode(view.notes)), moved=moved,
            notes=len(chart.notes), fit=round(grid_fit(chart), 4), hash=chart_hash(chart),
            bpm_changes=len(chart.tempo) > 1, fixed=sum(v for k, v in issues.count.items() if k.startswith("fixed")),
            tempo=[(str(b), v) for b, v in view.tempo],
        )
        if rec["level_value"] is None:
            rec.update(use=False, reason=f"level {c['level']!r} isn't a number")
        elif rec["fit"] < MIN_FIT:
            rec.update(use=False, reason=f"beats don't follow the music (fit {rec['fit']:.0%})")
        records.append(rec)
    return song, records
