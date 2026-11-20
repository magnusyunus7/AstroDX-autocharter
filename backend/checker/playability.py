"""Can two hands play this? Checks a simai note string (one BPM, like the
generator writes) for:
  * more notes at once than free hands (holds and slides keep a hand busy:
    a hold until it ends, a slide from its star tap until its travel ends)
  * a tap on a button that is being held
  * one hand alone (the other one busy) faster than ONE_HAND_SEC
Returns a list of problems (empty = fine). Touch notes count as a hand too.
"""
from __future__ import annotations

import re

from backend.simai.reader import note_groups

ONE_HAND_SEC = 0.149
_SLIDE = re.compile(r"[-<>^vpqszwV]")


def _dur_beats(tok: str) -> float:
    m = re.search(r"\[(\d+):(\d+)\]", tok)
    return 4.0 * int(m.group(2)) / int(m.group(1)) if m else 0.0


def check(body: str, first: float = 0.0) -> list[str]:
    m = re.search(r"\(([\d.]+)\)", body)
    if not m:
        return []
    beat = 60.0 / float(m.group(1))
    busy: list[tuple[float, float, int | None]] = []     # (start, end, button)
    alone: list[float] = []                               # notes played while a hand is busy
    problems: list[str] = []
    for t, toks in note_groups(body, first):
        busy = [b for b in busy if b[1] > t + 1e-6]
        if len(busy) + len(toks) > 2:
            problems.append(f"{t:.3f}s: {len(toks)} note(s) but {2 - len(busy)} free hand(s): {'/'.join(toks)}")
        for tok in toks:
            btn = int(tok[0]) if tok[0].isdigit() else None
            if btn and any(b[2] == btn for b in busy):
                problems.append(f"{t:.3f}s: {tok} on a held button")
        if busy and len(toks) == 1:
            if alone and alone[-1] > busy[0][0] and t - alone[-1] < ONE_HAND_SEC:
                problems.append(f"{t:.3f}s: one hand, {t - alone[-1]:.3f}s after the last note")
            alone.append(t)
        for tok in toks:
            btn = int(tok[0]) if tok[0].isdigit() else None
            if btn is None:
                continue
            if _SLIDE.search(tok[1:]):
                busy.append((t, t + beat + _dur_beats(tok) * beat, btn))
            elif "h" in re.sub(r"\[[^\]]*\]", "", tok):
                busy.append((t, t + _dur_beats(tok) * beat, btn))
    return problems