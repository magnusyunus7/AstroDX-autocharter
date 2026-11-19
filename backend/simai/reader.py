"""Read a simai note string into timed note groups (for statistics and
comparisons; not a full simai parser yet).

Handles (bpm) changes, {div} and {#seconds} commas, EACH with "/", the
two-digit tap shorthand ("15" = taps on 1 and 5), the "`" pseudo-EACH
separator, and the closing "E". Each returned group is one comma slot:
(time_seconds, [note tokens]).
"""
from __future__ import annotations

import re

_TWO_TAPS = re.compile(r"^[1-8]{2}$")


def note_groups(body: str, first: float = 0.0) -> list[tuple[float, list[str]]]:
    t, bpm, div, sec = first, None, 4.0, None
    out: list[tuple[float, list[str]]] = []
    token = []
    i, n = 0, len(body)

    def flush():
        raw = "".join(token).strip()
        token.clear()
        if not raw or raw == "E":
            return
        notes = []
        for part in re.split(r"[/`]", raw):
            part = part.strip()
            if not part:
                continue
            if _TWO_TAPS.match(part):
                notes += [part[0], part[1]]
            else:
                notes.append(part)
        if notes:
            out.append((t, notes))

    while i < n:
        c = body[i]
        if c == "(":
            j = body.index(")", i)
            bpm = float(body[i + 1:j]); i = j + 1
        elif c == "{":
            j = body.index("}", i)
            v = body[i + 1:j]
            if v.startswith("#"):
                sec, div = float(v[1:]), None
            else:
                div, sec = float(v), None
            i = j + 1
        elif c == "[":                       # durations: keep inside the token
            j = body.index("]", i)
            token.append(body[i:j + 1]); i = j + 1
        elif c == ",":
            flush()
            if sec is not None:
                t += sec
            elif bpm:
                t += 240.0 / bpm / div
            i += 1
        elif c in "\r\n\t ":
            i += 1
        else:
            token.append(c); i += 1
    flush()
    return out


def read_maidata(text: str) -> dict:
    """{'first': float, 'charts': {slot: body}} from a maidata.txt."""
    first = re.search(r"^&first=([-\d.]+)", text, re.M)
    charts = dict(re.findall(r"^&inote_(\d+)=(.*?)(?=^&|\Z)", text, re.M | re.S))
    return {"first": float(first.group(1)) if first else 0.0,
            "charts": {int(k): v for k, v in charts.items()}}