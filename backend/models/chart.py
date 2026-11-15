"""Internal chart representation.

All note times are in BEATS from chart start (beat 0 = bar 1, beat 1 = the
moment given by &first). We use Fraction so 1/3 or 1/6 beats stay exact and
the serializer can pick clean simai divisors.

A bar is always 4 beats (simai has no time signature; this is only for layout).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Union

BEATS_PER_BAR = 4

# Buttons are 1..8, clockwise, 1 = top-right.
BUTTONS = range(1, 9)

# Touch sensors: A1-A8, B1-B8, C (C1/C2 are aliases of C), D1-D8, E1-E8
TOUCH_GROUPS = ("A", "B", "D", "E")

# Slide shapes supported in v1 (simai symbols)
SLIDE_SHAPES = {"-", ">", "<", "^", "v", "p", "q", "s", "z", "pp", "qq", "w"}


def to_frac(x) -> Fraction:
    """Accept int/float/str/Fraction; floats are limited to 1/192 beat precision."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, float):
        return Fraction(x).limit_denominator(192)
    return Fraction(x)


def ring_distance(a: int, b: int) -> int:
    """Shortest distance between two buttons on the 8-button ring (0..4)."""
    d = abs(a - b) % 8
    return min(d, 8 - d)


@dataclass
class Tap:
    beat: Fraction
    button: int
    brk: bool = False      # BREAK
    ex: bool = False       # EX note
    star: bool = False     # star-shaped tap without a slide ($)

    def __post_init__(self):
        self.beat = to_frac(self.beat)


@dataclass
class Hold:
    beat: Fraction
    button: int
    length: Fraction       # in beats
    brk: bool = False
    ex: bool = False

    def __post_init__(self):
        self.beat = to_frac(self.beat)
        self.length = to_frac(self.length)

    @property
    def end(self) -> Fraction:
        return self.beat + self.length


@dataclass
class Slide:
    """Star tap at `beat` on `start`; the slide starts moving 1 beat later
    (simai default) and takes `length` beats to travel to `end`."""
    beat: Fraction
    start: int
    shape: str
    end_button: int
    length: Fraction       # travel time in beats
    brk: bool = False      # break star (head)
    head: bool = True      # False -> no star tap ("?" in simai)
    wait: Fraction = Fraction(1)  # fixed at 1 beat in v1 (simai default)

    def __post_init__(self):
        self.beat = to_frac(self.beat)
        self.length = to_frac(self.length)
        self.wait = to_frac(self.wait)

    @property
    def end(self) -> Fraction:
        return self.beat + self.wait + self.length


@dataclass
class Touch:
    beat: Fraction
    sensor: str            # "C", "B3", "E7", ...
    firework: bool = False

    def __post_init__(self):
        self.beat = to_frac(self.beat)


@dataclass
class TouchHold:
    beat: Fraction
    length: Fraction
    sensor: str = "C"
    firework: bool = False

    def __post_init__(self):
        self.beat = to_frac(self.beat)
        self.length = to_frac(self.length)

    @property
    def end(self) -> Fraction:
        return self.beat + self.length


Note = Union[Tap, Hold, Slide, Touch, TouchHold]


def note_end(n: Note) -> Fraction:
    return getattr(n, "end", n.beat)


@dataclass
class Chart:
    bpm: float
    notes: list[Note] = field(default_factory=list)

    def add(self, *notes: Note) -> "Chart":
        self.notes.extend(notes)
        return self

    def sorted_notes(self) -> list[Note]:
        return sorted(self.notes, key=lambda n: n.beat)

    @property
    def last_beat(self) -> Fraction:
        """Beat at which everything (incl. holds/slides) has finished."""
        return max((note_end(n) for n in self.notes), default=Fraction(0))

    def shifted(self, beats) -> "Chart":
        """Return a copy with every note moved by `beats` (used to add/remove
        leading empty bars)."""
        import copy
        b = to_frac(beats)
        out = Chart(self.bpm)
        for n in self.notes:
            m = copy.copy(n)
            m.beat = n.beat + b
            out.notes.append(m)
        return out


# ---------------------------------------------------------------- validation

class ChartError(ValueError):
    pass


def validate_note(n: Note) -> None:
    """Structural checks only (is this note legal simai?). Playability lives
    in checker/playability.py."""
    if n.beat < 0:
        raise ChartError(f"{n}: negative beat")
    if isinstance(n, (Tap, Hold)):
        if n.button not in BUTTONS:
            raise ChartError(f"{n}: button must be 1-8")
    if isinstance(n, (Hold, TouchHold)) and n.length <= 0:
        raise ChartError(f"{n}: length must be > 0")
    if isinstance(n, Slide):
        if n.start not in BUTTONS or n.end_button not in BUTTONS:
            raise ChartError(f"{n}: slide buttons must be 1-8")
        if n.shape not in SLIDE_SHAPES:
            raise ChartError(f"{n}: unsupported slide shape {n.shape!r}")
        if n.length <= 0:
            raise ChartError(f"{n}: slide length must be > 0")
        d = ring_distance(n.start, n.end_button)
        if n.shape == "-" and d < 2:
            raise ChartError(f"{n}: straight slide needs end at least 2 buttons away")
        if n.shape in ("^", "v") and d in (0, 4):
            raise ChartError(f"{n}: '{n.shape}' slide can't end on same/opposite button")
        if n.shape == "w" and d != 4:
            raise ChartError(f"{n}: wifi slide must end on the opposite button")
    if isinstance(n, (Touch, TouchHold)):
        s = n.sensor
        ok = s == "C" or (len(s) == 2 and s[0] in TOUCH_GROUPS and s[1] in "12345678")
        if not ok:
            raise ChartError(f"{n}: bad touch sensor {s!r}")
        if isinstance(n, TouchHold) and s != "C":
            raise ChartError(f"{n}: touch holds are C-only in v1")


def validate_chart(chart: Chart) -> None:
    if chart.bpm <= 0:
        raise ChartError("BPM must be > 0")
    for n in chart.notes:
        validate_note(n)
    # Two button notes starting on the same button at the same time is illegal.
    seen: set[tuple[Fraction, int]] = set()
    for n in chart.notes:
        btn = getattr(n, "button", None) or getattr(n, "start", None)
        if btn is None or (isinstance(n, Slide) and not n.head):
            continue
        key = (n.beat, btn)
        if key in seen:
            raise ChartError(f"two notes on button {btn} at beat {n.beat}")
        seen.add(key)