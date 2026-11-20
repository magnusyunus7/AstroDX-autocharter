"""Can two hands play this? Checks a simai note string (one BPM, like the
generator writes) for:
  * more notes at once than free hands (holds and slides keep a hand busy:
    a hold until it ends, a slide from its star tap until its travel ends)
  * a tap on a button that is being held
  * one hand alone (the other one busy) faster than ONE_HAND_SEC
  * a slide's hand is free while the star waits (1 beat) and busy while it
    travels. Notes at the moment it starts moving: one free hand, plus a tap on
    the slide's own start button (the slide hand taps and slides at once)
  * slide zone: from the moment it moves until slide_clear_beats (one {4},
    more at fast BPM) after it ends, nothing on or next to the slide's buttons
    (start, path, end), no touch notes, and (after it ends) no EACH: the slide
    would "splash" into them, or there's no time to get there. Exception:
    the slide hand's next note may come a {8} after it ends (more at fast BPM,
    slide_next_beats) if it goes right next to its end (not on the end, its
    path or its start): the shortest move. A chart doesn't say which hand plays
    what, so a note counts as the slide hand's when the other hand played less
    than ONE_HAND_SEC before it (the generator knows its hands exactly)
  * hold release: the held button gets the same room after the hold ends
  * straight and curve slides cover at least 2 buttons (1-3, 1<7; not 1-2, 1<8)
Returns a list of problems (empty = fine).
"""
from __future__ import annotations

import math
import re

from backend.simai.reader import note_groups

ONE_HAND_SEC = 0.149
FAST_SEC = 0.20                  # closer notes are played with alternating hands
CLEAR_SEC = 0.33                 # above ~180 BPM a {4} isn't enough room any more
CW_IS_RIGHT = {1, 2, 7, 8}       # simai: from these '>' turns clockwise; from 3-6 '<' does
_SLIDE = re.compile(r"^(\d)[bx$@?!]*([-<>^vpqszwV]+)(\d)")


def slide_clear_beats(bpm: float) -> float:
    """Room after a slide (or hold) ends: at least one {4} (1 beat); at fast
    BPM rounded up to half beats so it's at least CLEAR_SEC."""
    beat = 60.0 / bpm
    return max(1.0, math.ceil(CLEAR_SEC / beat * 2 - 1e-9) / 2)


def slide_next_beats(bpm: float) -> float:
    """Soonest the slide hand's next note may come after the slide ends (and
    then only right next to the end): a {8} (half a beat), more at fast BPM
    (half the finish room)."""
    return slide_clear_beats(bpm) / 2


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
    """Buttons kept free around a slide: its start, its path and end, and the
    buttons next to the path (where the moving star passes)."""
    path = slide_path(start, shape, end)
    return path | {start} | {(b % 8) + 1 for b in path} | {(b - 2) % 8 + 1 for b in path}


def next_spots(start: int, shape: str, end: int) -> set[int]:
    """Where the slide hand's next note may go right after the slide ends: next
    to the end, but not back on the slide's path (1>4 then 3 would be hit
    by the passing star) and not its start."""
    path = slide_path(start, shape, end)
    return {b for b in ((end % 8) + 1, (end - 2) % 8 + 1) if b not in path and b != start}


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
    soonest = slide_next_beats(bpm) * beat
    busy: list[tuple[float, float, int]] = []                   # (start, end, button) holds / slides
    zones: list[tuple] = []          # (moves, ends, ends + clear, kept-free buttons, name, start, end)
    released: list[tuple[float, float, int]] = []              # (hold end, end + clear, button)
    alone: list[float] = []                                     # notes played while a hand is busy
    last = None                                                 # previous note group
    problems: list[str] = []
    for t, toks in note_groups(body, first):
        busy = [b for b in busy if b[1] > t + 1e-6]
        zones = [z for z in zones if z[2] > t + 1e-6]
        released = [r for r in released if r[1] > t + 1e-6]
        moving = [b for b in busy if b[0] <= t + 1e-6]   # holds, and slides that have started moving
        # a tap on a slide's start button as it starts moving is that slide's hand
        own = [tok for tok in toks if tok[0].isdigit() and any(
            abs(z[0] - t) < 1e-6 and int(tok[0]) == z[5] for z in zones)]
        others = [tok for tok in toks if tok not in own]
        if len(moving) - len(own) + len(toks) > 2 or len(own) > sum(1 for z in zones if abs(z[0] - t) < 1e-6):
            problems.append(f"{t:.3f}s: {len(toks)} note(s) but {2 - len(moving)} free hand(s): {'/'.join(toks)}")
        for tok in others:
            btn = int(tok[0]) if tok[0].isdigit() else None
            if btn and any(b[2] == btn for b in moving):
                problems.append(f"{t:.3f}s: {tok} on a held button")
        for tok in toks:
            btn = int(tok[0]) if tok[0].isdigit() else None
            if btn and any(r[0] - 1e-6 <= t and r[2] == btn for r in released):
                problems.append(f"{t:.3f}s: {tok} right after a hold on that button ends")
        for z in zones:
            move, end, until, buttons, name, start, slide_end, spots = z[:8]
            if t < move - 1e-6:
                continue                                  # star still waiting: its hand is free
            after = t >= end - 1e-6
            when = f"{(t - end) / beat:.2f} beats after" if after else "during"
            if len(toks) > 1 and after:
                problems.append(f"{t:.3f}s: EACH {when} slide {name} ends")
            for tok in toks:
                if not tok[0].isdigit():
                    problems.append(f"{t:.3f}s: touch {tok} {when} slide {name}")
                    continue
                b = int(tok[0])
                if tok in own and abs(move - t) < 1e-6:
                    continue
                # which hand? the other hand plays everything while the slide moves;
                # afterwards a note is the slide hand's if the other hand couldn't
                # have played it (it played < ONE_HAND_SEC ago)
                other_free = t - z[9] >= ONE_HAND_SEC
                if not after or z[8]:
                    pass
                elif b in spots:
                    if t - end >= soonest - 1e-6:
                        z[8] = True                       # the slide hand's next note, right next to the end
                        continue
                    if not other_free:
                        z[8] = True
                        problems.append(f"{t:.3f}s: {tok} only {(t - end) / beat:.2f} beats after slide "
                                        f"{name} ends (the slide hand needs {soonest / beat:.2f})")
                        continue
                elif not other_free and b not in buttons:
                    z[8] = True
                    problems.append(f"{t:.3f}s: {tok} right after slide {name}: the slide hand's "
                                    f"next note goes next to its end ({'/'.join(map(str, sorted(spots)))})")
                    continue
                if b in buttons:
                    problems.append(f"{t:.3f}s: {tok} on/next to slide {name}, {when}")
        for z in zones:
            if t >= z[0] - 1e-6 and len(toks) == 1 and not z[8]:
                z[9] = t                                  # (still) the other hand's
        last = (t, toks)
        if moving and len(toks) == 1:
            if alone and alone[-1] > moving[0][0] and t - alone[-1] < ONE_HAND_SEC:
                problems.append(f"{t:.3f}s: one hand, {t - alone[-1]:.3f}s after the last note")
            alone.append(t)
        for tok in toks:
            sm = _SLIDE.match(tok)
            if sm:                                        # its hand is busy while it travels
                s, shape, e = int(sm.group(1)), sm.group(2), int(sm.group(3))
                span = min((e - s) % 8, (s - e) % 8) if shape == "-" else len(slide_path(s, shape, e))
                if shape in ("-", ">", "<", "^") and span < 2:
                    problems.append(f"{t:.3f}s: {tok}: a straight/curve slide must cover at least 2 buttons")
                move = t + beat
                end = move + _dur_beats(tok) * beat
                busy.append((move, end - 1e-6, s))
                zones.append([move, end - 1e-6, end + clear - 1e-6, slide_zone(s, shape, e),
                              f"{s}{shape}{e}", s, e, next_spots(s, shape, e), False, -1e9])
            elif tok[0].isdigit() and "h" in re.sub(r"\[[^\]]*\]", "", tok):
                end = t + _dur_beats(tok) * beat
                busy.append((t, end, int(tok[0])))
                released.append((end - 1e-6, end + clear - 1e-6, int(tok[0])))
    return problems