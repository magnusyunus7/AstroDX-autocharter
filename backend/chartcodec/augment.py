"""Rotate and mirror charts (Phase C1): each real chart gives 16 training
charts (8 rotations x mirrored or not). Both are exact relabellings: a
rotated or mirrored chart is just as playable as the original.

Rotation by r turns everything r buttons clockwise. Mirroring flips left and
right (button b -> 9 - b), which also turns clockwise into counter-clockwise.
"""
from __future__ import annotations

from dataclasses import replace

from backend.chartcodec.notes import Hold, Path, Segment, Slide, Tap, Touch, TouchHold

MIRROR_SHAPE = {"cw": "ccw", "ccw": "cw", "p": "q", "q": "p", "pp": "qq", "qq": "pp",
                "s": "z", "z": "s"}


def _btn(b: int, r: int, m: bool) -> int:
    if m:
        b = 9 - b
    return (b - 1 + r) % 8 + 1


def _sensor(s: str, r: int, m: bool) -> str:
    if s == "C":
        return s
    g, k = s[0], int(s[1])
    if m:
        # A/B sit at buttons (k -> 9-k); D/E sit between buttons k-1 and k (D1 at the top)
        k = 9 - k if g in "AB" else (10 - k - 1) % 8 + 1
    return f"{g}{(k - 1 + r) % 8 + 1}"


def transform_note(n, r: int = 0, mirror: bool = False):
    if isinstance(n, (Tap, Hold)):
        return replace(n, button=_btn(n.button, r, mirror))
    if isinstance(n, (Touch, TouchHold)):
        return replace(n, sensor=_sensor(n.sensor, r, mirror))
    if isinstance(n, Slide):
        paths = tuple(Path(tuple(Segment(MIRROR_SHAPE.get(s.shape, s.shape) if mirror else s.shape,
                                         _btn(s.end, r, mirror),
                                         None if s.mid is None else _btn(s.mid, r, mirror),
                                         s.length) for s in p.segments), p.wait, p.brk)
                      for p in n.paths)
        return replace(n, start=_btn(n.start, r, mirror), paths=paths)
    raise TypeError(f"unknown note {n!r}")


def transform(notes, r: int = 0, mirror: bool = False) -> list:
    """All notes rotated by r (0-7) buttons clockwise, mirrored first if asked."""
    return [transform_note(n, r % 8, mirror) for n in notes]
