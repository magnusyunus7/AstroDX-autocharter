"""Notes <-> token sequence: what the chart model reads and writes (Phase C1).

The chart is written left to right in time, like text:

    BEAT  P0 TAP B3 TAP B6  P96 SLIDE B1 PATH SH_- B5 L0 F96  BEAT ...

  * BEAT starts the next beat (every beat gets one, even an empty one, so the
    model always knows where it is in the bar);
  * Pk: the notes that follow sit k/192 of the way into the current beat;
    a tuplet position instead is Td Nn: n/d of the way (T5 N2 = 2/5 beat);
  * notes at the same moment are an EACH; PSEUDO starts the next step of a
    pseudo-EACH ("`");
  * a length is L<whole beats> followed by F<k/192> (or a tuplet Td Nn) when
    it has a fraction (L can repeat for lengths over 32 beats, they add up);
  * a slide is its star (SLIDE + button + marks), then each path: PATH,
    WAIT + length if the star waits other than 1 beat, PBRK for a break slide,
    then its pieces: SH_<shape> [turn button for V] end button [length].

Notes in one moment are always in the same order (notes.sort_key), so every
chart has exactly one token sequence. decode(encode(notes)) == notes for
notes whose values are all clean (notes.model_view makes sure of that).
"""
from __future__ import annotations

from fractions import Fraction

from backend.chartcodec.notes import (GRID, MAX_TUPLET, ONE, SENSORS, SHAPES, TUPLETS, Hold, Path,
                                      Segment, Slide, Tap, Touch, TouchHold, sort_key)

MAX_L = 32


def _vocab() -> list[str]:
    v = ["PAD", "BOS", "EOS", "BEAT", "PSEUDO"]
    v += [f"P{k}" for k in range(GRID)]
    v += ["TAP", "HOLD", "SLIDE", "TOUCH", "THOLD", "PATH", "WAIT"]
    v += [f"B{b}" for b in range(1, 9)]
    v += [f"S{s}" for s in SENSORS]
    v += ["BRK", "EX", "STAR1", "STAR2", "FW", "NOHEAD", "SUDDEN", "HEADTAP", "HEADPAD", "PBRK"]
    v += [f"SH_{s}" for s in SHAPES]
    v += [f"L{k}" for k in range(MAX_L + 1)]
    v += [f"F{k}" for k in range(1, GRID)]
    v += [f"T{d}" for d in TUPLETS]
    v += [f"N{k}" for k in range(1, MAX_TUPLET)]
    return v


VOCAB = _vocab()
INDEX = {t: i for i, t in enumerate(VOCAB)}


class TokenError(ValueError):
    pass


def _frac_tokens(f: Fraction, grid_prefix: str) -> list[str]:
    """Part of a beat (0 <= f < 1) -> [P/F k] on the grid, or [Td, Nn] for a tuplet."""
    d = f.denominator
    if GRID % d == 0:
        return [f"{grid_prefix}{int(f * GRID)}"]
    if d in TUPLETS:
        return [f"T{d}", f"N{f.numerator}"]
    raise TokenError(f"{f} of a beat is neither on the 1/{GRID} grid nor a tuplet (use notes.model_view)")


def length_tokens(x: Fraction) -> list[str]:
    x = Fraction(x)
    if x < 0:
        raise TokenError(f"negative length {x}")
    whole = x.numerator // x.denominator
    frac = x - whole
    out = []
    while whole > MAX_L:
        out.append(f"L{MAX_L}"); whole -= MAX_L
    out.append(f"L{whole}")
    if frac:
        out += _frac_tokens(frac, "F")
    return out


def _note_tokens(n) -> list[str]:
    if isinstance(n, Tap):
        t = ["TAP", f"B{n.button}"]
        if n.brk: t.append("BRK")
        if n.ex: t.append("EX")
        if n.star: t.append(f"STAR{n.star}")
        return t
    if isinstance(n, Hold):
        t = ["HOLD", f"B{n.button}"]
        if n.brk: t.append("BRK")
        if n.ex: t.append("EX")
        return t + length_tokens(n.length)
    if isinstance(n, Touch):
        return ["TOUCH", f"S{n.sensor}"] + (["FW"] if n.fw else []) + (["EX"] if n.ex else [])
    if isinstance(n, TouchHold):
        return (["THOLD", f"S{n.sensor}"] + (["FW"] if n.fw else []) + (["EX"] if n.ex else [])
                + length_tokens(n.length))
    if isinstance(n, Slide):
        t = ["SLIDE", f"B{n.start}"]
        if n.brk: t.append("BRK")
        if n.ex: t.append("EX")
        head = {"none": "NOHEAD", "sudden": "SUDDEN", "tap": "HEADTAP", "pad": "HEADPAD"}.get(n.head)
        if head: t.append(head)
        for p in n.paths:
            t.append("PATH")
            if p.wait != ONE:
                t += ["WAIT"] + length_tokens(p.wait)
            if p.brk:
                t.append("PBRK")
            for s in p.segments:
                t.append(f"SH_{s.shape}")
                if s.shape == "V":
                    t.append(f"B{s.mid}")
                t.append(f"B{s.end}")
                if s.length is not None:
                    t += length_tokens(s.length)
        return t
    raise TypeError(f"unknown note {n!r}")


def encode(notes) -> list[str]:
    """Notes (beats from the chart start) -> tokens, BOS ... EOS."""
    notes = sorted(notes, key=sort_key)
    out = ["BOS"]
    beat_no, pos, sub = -1, None, 0
    for n in notes:
        b = Fraction(n.beat)
        if b < 0:
            raise TokenError(f"note at negative beat {b}")
        whole = b.numerator // b.denominator
        k = b - whole
        while beat_no < whole:
            out.append("BEAT"); beat_no += 1; pos = None
        if k != pos:
            out += _frac_tokens(k, "P"); pos = k; sub = 0
        while sub < n.sub:
            out.append("PSEUDO"); sub += 1
        out += _note_tokens(n)
    out.append("EOS")
    return out


# ---------------------------------------------------------------- decode

class _Reader:
    def __init__(self, toks):
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, prefix=None):
        if self.i >= len(self.t):
            raise TokenError("sequence ends in the middle of a note")
        tok = self.t[self.i]
        if prefix is not None and not tok.startswith(prefix):
            raise TokenError(f"expected {prefix}... at {self.i}, got {tok}")
        self.i += 1
        return tok

    def flag(self, name) -> bool:
        if self.peek() == name:
            self.i += 1
            return True
        return False

    def button(self) -> int:
        tok = self.take("B")
        if tok not in INDEX:
            raise TokenError(f"bad button {tok}")
        return int(tok[1:])

    def numbered(self, prefix: str) -> bool:
        """Next token is <prefix><number> (F12, T5 ...; not FW, TAP ...)."""
        t = self.peek() or ""
        return t.startswith(prefix) and t[len(prefix):].isdigit() and t in INDEX

    def length(self, required=True):
        if not self.numbered("L"):
            if required:
                raise TokenError(f"expected a length at {self.i}, got {self.peek()}")
            return None
        units = 0
        while self.numbered("L"):
            units += int(self.take()[1:]) * GRID
        x = Fraction(units, GRID)
        if self.numbered("F"):
            x += Fraction(int(self.take()[1:]), GRID)
        elif self.numbered("T"):
            x += self.tuplet()
        return x

    def tuplet(self) -> Fraction:
        d = int(self.take("T")[1:])
        tok = self.take("N")
        if d not in TUPLETS or tok not in INDEX or not 0 < int(tok[1:]) < d:
            raise TokenError(f"bad tuplet T{d} {tok}")
        return Fraction(int(tok[1:]), d)


def decode(tokens) -> list:
    """Tokens -> notes. Raises TokenError on a sequence that isn't a chart
    (useful when sampling from the model: it tells exactly what went wrong)."""
    r = _Reader([t for t in tokens if t not in ("PAD",)])
    notes = []
    beat_no, pos, sub = -1, None, 0
    if r.peek() == "BOS":
        r.take()
    while r.peek() is not None:
        tok = r.take()
        if tok == "EOS":
            break
        if tok == "BEAT":
            beat_no += 1; pos = None; continue
        if tok[0] == "P" and tok[1:].isdigit():
            if tok not in INDEX:
                raise TokenError(f"bad position {tok}")
            pos, sub = Fraction(int(tok[1:]), GRID), 0; continue
        if tok[0] == "T" and tok[1:].isdigit():
            r.i -= 1
            pos, sub = r.tuplet(), 0; continue
        if tok == "PSEUDO":
            if pos is None:
                raise TokenError("PSEUDO before a position")
            sub += 1; continue
        if pos is None or beat_no < 0:
            raise TokenError(f"note {tok} before BEAT / position at {r.i - 1}")
        beat = beat_no + pos
        if tok == "TAP":
            b = r.button(); brk = r.flag("BRK"); ex = r.flag("EX")
            star = 1 if r.flag("STAR1") else 2 if r.flag("STAR2") else 0
            notes.append(Tap(beat, b, brk, ex, star, sub))
        elif tok == "HOLD":
            b = r.button(); brk = r.flag("BRK"); ex = r.flag("EX")
            notes.append(Hold(beat, b, r.length(), brk, ex, sub))
        elif tok == "TOUCH":
            s = r.take("S")[1:]; fw = r.flag("FW"); ex = r.flag("EX")
            if s not in SENSORS:
                raise TokenError(f"bad sensor {s}")
            notes.append(Touch(beat, s, fw, ex, sub))
        elif tok == "THOLD":
            s = r.take("S")[1:]; fw = r.flag("FW"); ex = r.flag("EX")
            if s not in SENSORS:
                raise TokenError(f"bad sensor {s}")
            notes.append(TouchHold(beat, s, r.length(), fw, ex, sub))
        elif tok == "SLIDE":
            start = r.button(); brk = r.flag("BRK"); ex = r.flag("EX")
            head = ("none" if r.flag("NOHEAD") else "sudden" if r.flag("SUDDEN") else
                    "tap" if r.flag("HEADTAP") else "pad" if r.flag("HEADPAD") else "star")
            paths = []
            while r.flag("PATH"):
                wait = r.length() if r.flag("WAIT") else ONE
                pbrk = r.flag("PBRK")
                segs = []
                while (r.peek() or "").startswith("SH_"):
                    shape = r.take()[3:]
                    if shape not in SHAPES:
                        raise TokenError(f"bad shape {shape}")
                    mid = r.button() if shape == "V" else None
                    end = r.button()
                    segs.append(Segment(shape, end, mid, r.length(required=False)))
                if not segs or segs[-1].length is None:
                    raise TokenError(f"slide path without pieces / final length at {r.i}")
                paths.append(Path(tuple(segs), wait, pbrk))
            if not paths:
                raise TokenError(f"slide without a path at {r.i}")
            notes.append(Slide(beat, start, tuple(paths), head, brk, ex, sub))
        else:
            raise TokenError(f"unexpected token {tok} at {r.i - 1}")
    return sorted(notes, key=sort_key)


def ids(tokens) -> list[int]:
    return [INDEX[t] for t in tokens]


def from_ids(seq) -> list[str]:
    return [VOCAB[i] for i in seq]
