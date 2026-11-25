"""The full note model used by the chart model (Phase C1).

Everything a simai chart says about notes, in one place:
  * times are BEATS from the chart start (beat 0 = the moment given by &first),
    counted by the chart's own commas, so a BPM change doesn't break the count;
  * every time and length is kept EXACTLY as the chart says (Fractions), even
    odd ones ({625} fillers from old converted charts, seconds-based values);
  * the model only works with "clean" values: the 1/192-beat grid ({768} and
    everything coarser) plus tuplets with a small denominator (5ths, 7ths,
    9ths, 10ths ... of a beat). model_view() gives the model a copy of the
    notes where any other value is moved to the nearest grid point (a few ms
    at most); the chart itself is never changed;
  * lengths of holds / slides are in beats of the BPM at the note's start
    (that's what simai's [N:M] means);
  * slide directions are stored as cw / ccw, not simai's "<" / ">" (whose
    meaning flips between the top and bottom half of the ring), so rotating
    or mirroring a chart is a plain relabelling.

Buttons 1..8 go clockwise from the top right. Touch sensors: A1-A8 and B1-B8
sit at the buttons, D1-D8 and E1-E8 sit between buttons (D1/E1 at the top,
between 8 and 1), C is the centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Union

GRID = 192                      # grid points per beat: {768} and every coarser divisor
MAX_TUPLET = 48


def _musical(d: int) -> bool:
    for p in (2, 3, 5, 7):
        while d % p == 0:
            d //= p
    return d == 1


# Tuplets kept exact: 5ths, 7ths, 9ths, 10ths ... of a beat (only 2, 3, 5, 7 in
# the denominator). 11ths, 13ths, 17ths ... aren't rhythms charters write; in
# the collection they come from old converted charts squeezing tempo changes
# into odd divisors, so the model sees those at the nearest grid point.
TUPLETS = tuple(d for d in range(2, MAX_TUPLET + 1) if GRID % d and _musical(d))
ONE = Fraction(1)

# Slide shapes after parsing. cw/ccw replace simai's "<" / ">".
SHAPES = ("-", "^", "v", "cw", "ccw", "p", "q", "pp", "qq", "s", "z", "V", "w")
SENSOR_GROUPS = "ABDE"
SENSORS = ("C",) + tuple(f"{g}{i}" for g in SENSOR_GROUPS for i in range(1, 9))


def snap(x: Fraction) -> Fraction:
    """Nearest 1/GRID-beat grid point."""
    return Fraction(round(Fraction(x) * GRID), GRID)


def clean(x: Fraction) -> bool:
    """On the grid, or a small tuplet (its part within the beat is n/d, d <= MAX_TUPLET)."""
    f = Fraction(x) - (Fraction(x).numerator // Fraction(x).denominator)
    return GRID % f.denominator == 0 or f.denominator in TUPLETS


def model_value(x: Fraction) -> Fraction:
    return x if clean(x) else snap(x)


@dataclass(frozen=True)
class Tap:
    beat: Fraction
    button: int
    brk: bool = False
    ex: bool = False
    star: int = 0               # 1: "$" star-shaped tap, 2: "$$"
    sub: int = 0                # position in a pseudo-EACH ("`") run; 0 = normal


@dataclass(frozen=True)
class Hold:
    beat: Fraction
    button: int
    length: Fraction            # beats (0 = "1h" with no length)
    brk: bool = False
    ex: bool = False
    sub: int = 0


@dataclass(frozen=True)
class Touch:
    beat: Fraction
    sensor: str
    fw: bool = False            # firework ("f")
    ex: bool = False
    sub: int = 0


@dataclass(frozen=True)
class TouchHold:
    beat: Fraction
    sensor: str
    length: Fraction
    fw: bool = False
    ex: bool = False
    sub: int = 0


@dataclass(frozen=True)
class Segment:
    """One piece of a slide path: from the previous button to `end`.
    `mid` is the turning button of a grand V ("1V35": mid 3, end 5).
    `length` is this piece's own travel time in beats, or None when the
    path gives one total time at its end ("1-4-7[4:1]")."""
    shape: str
    end: int
    mid: Optional[int] = None
    length: Optional[Fraction] = None


@dataclass(frozen=True)
class Path:
    """One slide from the star: one or more segments. The star waits `wait`
    beats before it moves (simai's default is 1 beat)."""
    segments: tuple[Segment, ...]
    wait: Fraction = ONE
    brk: bool = False           # break slide (the body, not the star)


@dataclass(frozen=True)
class Slide:
    """A star on `start` with one or more slide paths ("1-5[8:1]*-3[8:1]")."""
    beat: Fraction
    start: int
    paths: tuple[Path, ...]
    head: str = "star"          # "star", "none" ("?" no star tap), "sudden" ("!" star not shown),
                                #  "tap" ("@" plain tap instead of a star), "pad" ("$" maiPad tapless)
    brk: bool = False           # break star
    ex: bool = False
    sub: int = 0


Note = Union[Tap, Hold, Touch, TouchHold, Slide]
KIND_ORDER = {Tap: 0, Hold: 1, Slide: 2, Touch: 3, TouchHold: 4}


def where(n: Note) -> str:
    """Button or sensor of a note, as text (for sorting and messages)."""
    if isinstance(n, (Tap, Hold)):
        return str(n.button)
    if isinstance(n, Slide):
        return str(n.start)
    return n.sensor


def sort_key(n: Note):
    """Canonical order: by time, pseudo-EACH position, kind, place, then the
    rest of the note (so equal charts always list notes the same way)."""
    return (n.beat, n.sub, KIND_ORDER[type(n)], where(n), repr(n))


@dataclass
class ChartData:
    """One difficulty: its tempo map and notes."""
    tempo: list[tuple[Fraction, float]] = field(default_factory=list)   # (beat, bpm), first at beat 0
    notes: list[Note] = field(default_factory=list)

    def canonical(self) -> "ChartData":
        return ChartData(list(self.tempo), sorted(self.notes, key=sort_key))

    def seconds(self, beat: Fraction) -> float:
        """Seconds from the chart start (&first) to `beat`, following the tempo map."""
        t, prev_b, prev_v = 0.0, Fraction(0), self.tempo[0][1]
        for b, v in self.tempo:
            if b >= beat:
                break
            t += float(b - prev_b) * 60.0 / prev_v
            prev_b, prev_v = b, v
        return t + float(Fraction(beat) - prev_b) * 60.0 / prev_v

    def model_view(self) -> tuple["ChartData", int]:
        """The chart as the model sees it: notes through model_view() and
        tempo changes on odd beats moved to the nearest grid point too."""
        notes, moved = model_view(self.notes)
        tempo = []
        for b, v in self.tempo:
            nb = model_value(b)
            moved += nb != b
            if tempo and tempo[-1][0] == nb:
                tempo[-1] = (nb, v)
            else:
                tempo.append((nb, v))
        return ChartData(tempo, notes), moved

    def bpm_at(self, beat: Fraction) -> float:
        bpm = self.tempo[0][1]
        for b, v in self.tempo:
            if b <= beat:
                bpm = v
        return bpm


def model_view(notes) -> tuple[list, int]:
    """Copy of the notes the model can read: every time / length / wait that
    isn't clean() moved to the nearest grid point. Returns (notes, how many
    values moved). The chart itself is untouched."""
    from dataclasses import replace
    moved = 0

    def mv(x):
        nonlocal moved
        if x is None:
            return None
        y = model_value(x)
        moved += y != x
        return y

    out = []
    for n in notes:
        if isinstance(n, Slide):
            paths = tuple(Path(tuple(Segment(s.shape, s.end, s.mid, mv(s.length)) for s in p.segments),
                               mv(p.wait), p.brk) for p in n.paths)
            out.append(replace(n, beat=mv(n.beat), paths=paths))
        elif isinstance(n, (Hold, TouchHold)):
            out.append(replace(n, beat=mv(n.beat), length=mv(n.length)))
        else:
            out.append(replace(n, beat=mv(n.beat)))
    return sorted(out, key=sort_key), moved


def same_notes(a, b) -> bool:
    """Equal note lists, with slide waits compared to 1e-9 beat (the writer
    puts waits in seconds, which can't always be exact)."""
    from dataclasses import replace

    def norm(n):
        if isinstance(n, Slide):
            return replace(n, paths=tuple(replace(p, wait=Fraction(round(p.wait * 10**9), 10**9))
                                          for p in n.paths))
        return n
    return len(a) == len(b) and all(norm(x) == norm(y) for x, y in zip(a, b))
