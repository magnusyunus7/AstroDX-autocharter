"""Chart -> simai note text.

Output style (made for editing in MajdataEdit):
  * first line starts with (BPM)
  * one bar per line, each line starts with its own {divisor}
  * empty bar -> "{1},"
  * no comments

simai timing recap: {N} means every comma is 1/N of a whole note (= 4/N beats).
Durations use [N:M] = M x (1/N whole note).
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import lcm

from backend.models.chart import (
    BEATS_PER_BAR, Chart, Hold, Note, Slide, Tap, Touch, TouchHold, validate_chart,
)

MAX_DIVISOR = 384  # safety cap; anything finer than this is almost surely a bug
READABLE_BAR_DIVISOR = 16  # above this, split the bar into per-beat divisors


def fmt_number(x: float) -> str:
    """180.0 -> '180', 174.5 -> '174.5'"""
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s or "0"


def fmt_duration(beats: Fraction) -> str:
    """Beats -> simai [N:M]. 1 beat -> [4:1], 1.5 beats -> [8:3]."""
    whole = Fraction(beats) / BEATS_PER_BAR  # fraction of a whole note
    return f"[{whole.denominator}:{whole.numerator}]"


def note_token(n: Note) -> str:
    if isinstance(n, Tap):
        t = str(n.button)
        if n.brk:
            t += "b"
        if n.ex:
            t += "x"
        if n.star:
            t += "$"
        return t
    if isinstance(n, Hold):
        t = str(n.button)
        if n.brk:
            t += "b"
        if n.ex:
            t += "x"
        return t + "h" + fmt_duration(n.length)
    if isinstance(n, Slide):
        t = str(n.start)
        if n.brk:
            t += "b"
        if not n.head:
            t += "?"
        return f"{t}{n.shape}{n.end_button}{fmt_duration(n.length)}"
    if isinstance(n, Touch):
        return n.sensor + ("f" if n.firework else "")
    if isinstance(n, TouchHold):
        return n.sensor + "h" + ("f" if n.firework else "") + fmt_duration(n.length)
    raise TypeError(f"unknown note {n!r}")


def _bar_divisor(offsets: list[Fraction]) -> int:
    """Smallest simai divisor (multiple of 4 when the bar has notes, so the
    grid stays readable) that puts every note offset on a comma."""
    if not offsets:
        return 1
    d = 4
    for off in offsets:
        # slot = off * d / 4 must be an integer -> d must be multiple of 4/gcd stuff
        need = (off / BEATS_PER_BAR).denominator
        d = lcm(d, need)
    if d > MAX_DIVISOR:
        raise ValueError(f"bar needs divisor {d} (> {MAX_DIVISOR}); offsets={offsets}")
    return d


def serialize(chart: Chart, end_beat: Fraction | None = None) -> str:
    """Return the note text for one difficulty (what goes after &inote_N=),
    including the final E."""
    validate_chart(chart)

    by_beat: dict[Fraction, list[Note]] = defaultdict(list)
    for n in chart.notes:
        by_beat[n.beat].append(n)

    last = chart.last_beat if end_beat is None else Fraction(end_beat)
    # number of bars: enough to contain every note start and every hold/slide end
    n_bars = max(int(-(-last // BEATS_PER_BAR)), 1)  # ceil(last / 4)
    # make sure the bar containing the last note *start* exists
    if by_beat:
        n_bars = max(n_bars, int(max(by_beat) // BEATS_PER_BAR) + 1)

    lines: list[str] = []
    for bar in range(n_bars):
        bar_start = Fraction(bar * BEATS_PER_BAR)
        beats_here = sorted(b for b in by_beat if bar_start <= b < bar_start + BEATS_PER_BAR)
        offsets = [b - bar_start for b in beats_here]
        div = _bar_divisor(offsets)
        if div <= READABLE_BAR_DIVISOR:
            line = _render_span(by_beat, bar_start, BEATS_PER_BAR, div)
        else:
            line = _render_bar_per_beat(by_beat, bar_start)
        if bar == 0:
            line = f"({fmt_number(chart.bpm)})" + line
        lines.append(line)

    lines.append("E")
    return "\n".join(lines)


def _render_span(by_beat, start: Fraction, length_beats: int, div: int,
                 prev_div: int | None = None) -> str:
    """Render `length_beats` beats starting at `start` with divisor `div`.
    Omits the {div} tag when it equals prev_div."""
    step = Fraction(BEATS_PER_BAR, div)
    n_slots = int(Fraction(length_beats) / step)
    slots = [""] * n_slots
    for b in by_beat:
        if start <= b < start + length_beats:
            idx = (b - start) / step
            assert idx.denominator == 1, (b, div)
            group = sorted(by_beat[b], key=_sort_key)
            slots[int(idx)] = "/".join(note_token(n) for n in group)
    tag = "" if div == prev_div else f"{{{div}}}"
    return tag + "".join(s + "," for s in slots)


def _render_bar_per_beat(by_beat, bar_start: Fraction) -> str:
    """Bar with mixed rhythms (e.g. 16ths + triplets): pick a divisor per
    beat so the line reads like {16}1,2,3,4,{12}5,6,7, instead of {48}..."""
    out, prev = [], None
    for i in range(BEATS_PER_BAR):
        s = bar_start + i
        offs = [b - s for b in by_beat if s <= b < s + 1]
        d = _bar_divisor(offs) if offs else 4
        out.append(_render_span(by_beat, s, 1, d, prev))
        prev = d
    return "".join(out)


def _sort_key(n: Note):
    # stable, readable ordering inside an EACH: buttons ascending, touches last
    btn = getattr(n, "button", None) or getattr(n, "start", None)
    return (0, btn) if btn is not None else (1, getattr(n, "sensor", ""))