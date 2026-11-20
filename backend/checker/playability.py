"""Can two hands play this? Checks a simai note string (one BPM, like the
generator writes) for:
  * more notes at once than free hands (holds and slides keep a hand busy:
    a hold until it ends, a slide from its star tap until its travel ends)
  * a tap on a button that is being held
  * one hand alone (the other one busy) faster than ONE_HAND_SEC
  * slide zone: from the star until slide_clear_beats (one {4}, more at fast
    BPM) after the slide ends, nothing on or next to the slide's buttons
    (start, path, end), no touch notes, and (after it ends) no EACH: the
    slide would "splash" into them, or there's no time to get there
  * hold release: the held button gets the same room after the hold ends
Returns a list of problems (empty = fine).
"""
from __future__ import annotations

import math
import re

from backend.simai.reader import note_groups

ONE_HAND_SEC = 0.149
CLEAR_SEC = 0.33                 # above ~180 BPM a {4} isn't enough room any more
CW_IS_RIGHT = {1, 2, 7, 8}       # simai: from these '>' turns clockwise; from 3-6 '<' does
_SLIDE = re.compile(r"^(\d)[bx$@?!]*([-<>^vpqszwV]+)(\d)")


def slide_clear_beats(bpm: float) -> float:
    """Room after a slide (or hold) ends: at least one {4} (1 beat); at fast
    BPM rounded up to half beats so it's at least CLEAR_SEC."""
    beat = 60.0 / bpm
    return max(1.0, math.ceil(CLEAR_SEC / beat * 2 - 1e-9) / 2)


def slide_path(start: int, shape: str, end: int) -> set[int]:
    """Buttons a slide finishes over or passes: its end, and the buttons along
    an arc. Straight/v/loops cross the middle, so just the end."""
    path = {end}
    if shape in (">", "<", "^"):
        if shape == "^":
            cw = (end - start) % 8 <= 4
        else:
            cw = (shape == ">") == (start in CW_IS_RIGHT)
        k = (end - start) % 8 if cw else (start - end) % 8
        path |= {(start - 1 + (s if cw else -s)) % 8 + 1 for s in range(1, k + 1)}
    return path


def slide_zone(start: int, shape: str, end: int) -> set[int]:
    """Buttons kept free around a slide: start, path, end and their neighbours."""
    core = slide_path(start, shape, end) | {start}
    return core | {(b % 8) + 1 for b in core} | {(b - 2) % 8 + 1 for b in core}


def _dur_beats(tok: str) -> float:
    m = re.search(r"\[(\d+):(\d+)\]", tok)
    return 4.0 * int(m.group(2)) / int(m.group(1)) if m else 0.0


def check(body: str, first: float = 0.0) -> list[str]:
    m = re.search(r"\(([\d.]+)\)", body)
    if not m:
        return []
    bpm = float(m.group(1))
    beat = 60.0 / bpm
    clear = slide_clear_beats(bpm) * beat
    busy: list[tuple[float, float, int]] = []                   # (start, end, button) holds / slides
    zones: list[tuple[float, float, float, set[int], str]] = []  # (star, end, end + clear, buttons, slide)
    released: list[tuple[float, float, int]] = []              # (hold end, end + clear, button)
    alone: list[float] = []                                     # notes played while a hand is busy
    problems: list[str] = []
    for t, toks in note_groups(body, first):
        busy = [b for b in busy if b[1] > t + 1e-6]
        zones = [z for z in zones if z[2] > t + 1e-6]
        released = [r for r in released if r[1] > t + 1e-6]
        if len(busy) + len(toks) > 2:
            problems.append(f"{t:.3f}s: {len(toks)} note(s) but {2 - len(busy)} free hand(s): {'/'.join(toks)}")
        for tok in toks:
            btn = int(tok[0]) if tok[0].isdigit() else None
            if btn and any(b[2] == btn for b in busy):
                problems.append(f"{t:.3f}s: {tok} on a held button")
            if btn and any(r[0] - 1e-6 <= t and r[2] == btn for r in released):
                problems.append(f"{t:.3f}s: {tok} right after a hold on that button ends")
        for star, end, until, buttons, name in zones:
            if t <= star + 1e-6:
                continue
            when = f"{(t - end) / beat:.2f} beats after" if t >= end - 1e-6 else "during"
            if len(toks) > 1 and t >= end - 1e-6:
                problems.append(f"{t:.3f}s: EACH {when} slide {name} ends")
            for tok in toks:
                if not tok[0].isdigit():
                    problems.append(f"{t:.3f}s: touch {tok} {when} slide {name}")
                elif int(tok[0]) in buttons:
                    problems.append(f"{t:.3f}s: {tok} on/next to slide {name}, {when}")
        if busy and len(toks) == 1:
            if alone and alone[-1] > busy[0][0] and t - alone[-1] < ONE_HAND_SEC:
                problems.append(f"{t:.3f}s: one hand, {t - alone[-1]:.3f}s after the last note")
            alone.append(t)
        for tok in toks:
            sm = _SLIDE.match(tok)
            if sm:                                        # busy from the star to the end of travel
                s, shape, e = int(sm.group(1)), sm.group(2), int(sm.group(3))
                end = t + beat + _dur_beats(tok) * beat
                busy.append((t, end - 1e-6, s))
                zones.append((t, end - 1e-6, end + clear - 1e-6, slide_zone(s, shape, e), f"{s}{shape}{e}"))
            elif tok[0].isdigit() and "h" in re.sub(r"\[[^\]]*\]", "", tok):
                end = t + _dur_beats(tok) * beat
                busy.append((t, end, int(tok[0])))
                released.append((end - 1e-6, end + clear - 1e-6, int(tok[0])))
    return problems