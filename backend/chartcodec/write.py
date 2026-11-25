"""ChartData -> simai text (Phase C1).

Writes what parse.py reads, in one fixed style:
  * one bar (4 beats) per line, each starting with its {divisor}; a bar with
    mixed rhythms (e.g. 16ths and triplets) gets a {divisor} per beat;
  * a chart with odd exact times (old converted charts) is written event by
    event, each gap with its own {divisor}(s), so every time stays exact
    (WriteError if that would need absurd divisors: write its model_view());
  * (bpm) at the slot where the tempo changes;
  * lengths as [N:M]; a slide whose star waits other than 1 beat as
    [seconds##N:M] in its first bracket;
  * "<" / ">" picked from cw / ccw and the segment's start button.
parse_chart(write_chart(c)) gives the same chart for every chart parse_chart
produces (slide waits match to 1e-9 beat: they're written in seconds).
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import lcm

from backend.chartcodec.notes import (ONE, ChartData, Hold, Path, Slide, Tap, Touch, TouchHold,
                                      sort_key)

BAR = 4
READABLE = 16                    # bars needing a finer {divisor} are written beat by beat
MAX_DIV = 768                    # finer than {768}: the bar is written gap by gap
MAX_ODD_DIV = 10 ** 7            # odd times needing finer divisors than this can't be written
MAX_COMMAS = 4096                # nor gaps needing more commas than this


class WriteError(ValueError):
    """The chart has exact times too odd to write as simai text (only happens
    with old converted charts whose tempo changes were squeezed into odd
    divisors; their model_view() writes fine)."""


def fmt_bpm(v: float) -> str:
    return f"{v:.10g}"


def fmt_len(beats: Fraction) -> str:
    whole = Fraction(beats) / BAR
    return f"{whole.denominator}:{whole.numerator}"


def _symbol(shape: str, start: int) -> str:
    if shape not in ("cw", "ccw"):
        return shape
    top = start in (1, 2, 7, 8)
    return ">" if (shape == "cw") == top else "<"


def _path(start: int, p: Path, bpm: float) -> str:
    out, prev, first_bracket = [], start, True
    for seg in p.segments:
        out.append(_symbol(seg.shape, prev))
        if seg.shape == "V":
            out.append(str(seg.mid))
        out.append(str(seg.end))
        if seg.length is not None:
            if first_bracket and p.wait != ONE:
                sec = float(p.wait) * 60.0 / bpm
                out.append(f"[{sec!r}##{fmt_len(seg.length)}]")
            else:
                out.append(f"[{fmt_len(seg.length)}]")
            first_bracket = False
        prev = seg.end
    if p.brk:
        out.append("b")
    return "".join(out)


def note_text(n, bpm: float) -> str:
    if isinstance(n, Tap):
        return f"{n.button}{'b' * n.brk}{'x' * n.ex}{'$' * n.star}"
    if isinstance(n, Hold):
        return f"{n.button}{'b' * n.brk}{'x' * n.ex}h" + (f"[{fmt_len(n.length)}]" if n.length else "")
    if isinstance(n, Touch):
        return n.sensor + "f" * n.fw + "x" * n.ex
    if isinstance(n, TouchHold):
        return n.sensor + "x" * n.ex + "h" + "f" * n.fw + (f"[{fmt_len(n.length)}]" if n.length else "")
    if isinstance(n, Slide):
        head = {"star": "", "none": "?", "sudden": "!", "tap": "@", "pad": "$"}[n.head]
        return (f"{n.start}{'b' * n.brk}{'x' * n.ex}{head}"
                + "*".join(_path(n.start, p, bpm) for p in n.paths))
    raise TypeError(f"unknown note {n!r}")


def _slot_text(notes, bpm: float) -> str:
    """Notes of one slot: '/' inside a pseudo-EACH step, '`' between steps."""
    steps = defaultdict(list)
    for n in sorted(notes, key=sort_key):
        steps[n.sub].append(note_text(n, bpm))
    return "`".join("/".join(steps[s]) for s in sorted(steps))


def _divisor(offsets, span: int) -> int:
    """Smallest {d} (a multiple of 4 / span... i.e. whole slots per span) that
    puts every offset (beats from the span start) on a comma."""
    d = 4 // span if span < 4 else 1
    d = max(d, 1)
    for off in offsets:
        d = lcm(d, (Fraction(off) / BAR).denominator)
    if span == 1:
        d = lcm(d, 4)
    return d


def _gap_pieces(g: Fraction) -> list[tuple[int, int]]:
    """A gap in whole notes -> [(divisor, commas)] adding up to it exactly.
    Clean gaps are commas of one divisor. An odd gap p/q is split by its
    denominator: q = q1 * q2 with q1 made of 2s and 3s, q2 the rest, and
    p/q = u/q1 + v/q2 -> u commas of {q1} then v commas of {q2}."""
    p, q = g.numerator, g.denominator
    q1 = 1
    for f in (2, 3):
        while q % (q1 * f) == 0:
            q1 *= f
    q2 = q // q1
    if p <= MAX_COMMAS and q <= MAX_ODD_DIV:                    # p commas of {q}
        return [(q, p)]
    if q2 == 1 or q1 == 1:
        raise WriteError(f"a gap of {g} whole notes can't be written compactly in simai")
    v = (p * pow(q1, -1, q2)) % q2                               # p = u*q2 + v*q1
    u = (p - v * q1) // q2
    if u < 0 or u > MAX_COMMAS or v > MAX_COMMAS or q2 > MAX_ODD_DIV:
        raise WriteError(f"a gap of {g} whole notes can't be written compactly in simai")
    return [(d, c) for d, c in ((q1, u), (q2, v)) if c]


def _write_events(chart, notes_at, tempo_at, events) -> str:
    """Charts with odd exact times (old converted charts): written event by
    event, each gap to the next event with its own divisor(s). Notes after an
    odd filler share the same small offset, so the gaps between them are
    clean again; only the filler itself needs odd divisors."""
    lines, line, prev_div = [], [], None
    bpm_now = chart.tempo[0][1]
    here = [Fraction(0)] + [b for b in events if b > 0]
    for a, b in zip(here, here[1:] + [None]):
        if a in tempo_at:
            bpm_now = tempo_at[a]
            line.append(f"({fmt_bpm(bpm_now)})")
        text = _slot_text(notes_at.get(a, ()), bpm_now)
        if b is None:                                            # last event: one comma ends it
            line.append(text + ",")
            break
        for d, c in _gap_pieces((b - a) / BAR):
            if d != prev_div:
                line.append(f"{{{d}}}")
                prev_div = d
            line.append(text + "," * c)
            text = ""
        if b // BAR != a // BAR:                                 # crossed a bar line: new line
            lines.append("".join(line)); line = []
    if line:
        lines.append("".join(line))
    lines.append("E")
    return "\n".join(lines)


def write_chart(chart: ChartData) -> str:
    """The note text for one difficulty (what follows &inote_N=), ending in E."""
    notes_at = defaultdict(list)
    for n in chart.notes:
        notes_at[n.beat].append(n)
    tempo_at = {b: v for b, v in chart.tempo}
    events = set(notes_at) | set(tempo_at)
    last = max(events, default=Fraction(0))
    n_bars = int(last // BAR) + 1

    bpm_now = chart.tempo[0][1]
    lines, prev_div = [], None

    def span(start: Fraction, length: int, div: int) -> str:
        nonlocal bpm_now, prev_div
        step = Fraction(BAR, div)
        parts = []
        for k in range(int(length / step)):
            b = start + k * step
            if b in tempo_at:
                bpm_now = tempo_at[b]
                parts.append(f"({fmt_bpm(bpm_now)})")
            if k == 0 and div != prev_div:
                parts.append(f"{{{div}}}")
            parts.append(_slot_text(notes_at.get(b, ()), bpm_now) + ",")
        prev_div = div
        return "".join(parts)

    for bar in range(n_bars):
        s = Fraction(bar * BAR)
        offs = [b - s for b in events if s <= b < s + BAR]
        d = _divisor(offs, BAR)
        if d <= READABLE:
            lines.append(span(s, BAR, d))
            continue
        per_beat = [_divisor([b - (s + k) for b in events if s + k <= b < s + k + 1], 1) for k in range(BAR)]
        if max(per_beat) > MAX_DIV:
            return _write_events(chart, notes_at, tempo_at, sorted(events))
        lines.append("".join(span(s + k, 1, per_beat[k]) for k in range(BAR)))
    lines.append("E")
    return "\n".join(lines)
