"""Which token may come next (Phase C3/C4): the C1 token grammar as a state
machine, so a chart written by the model always decodes, plus a few rules
no real chart breaks:
  * time only moves forward (a new position must be later in the beat);
  * one note per button / sensor per moment;
  * slide shapes that fit their buttons (1-3 yes, 1-2 no; s/z/w only to the
    opposite button; a grand V turns 2 buttons from its start ...).
Playability across moments (hands, slide zones) is the checker's job, not
this file's.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from backend.chartcodec.notes import GRID, SENSORS, SHAPES, TUPLETS
from backend.chartcodec.tokens import INDEX, MAX_L, VOCAB

V = len(VOCAB)


def _ids(names):
    return np.array([INDEX[n] for n in names], dtype=np.int64)


BUTTONS = [f"B{b}" for b in range(1, 9)]
SENS = [f"S{s}" for s in SENSORS]
L_TOK = [f"L{k}" for k in range(MAX_L + 1)]
F_TOK = [f"F{k}" for k in range(1, GRID)]
T_TOK = [f"T{d}" for d in TUPLETS]


def ring(a: int, b: int) -> int:
    d = abs(a - b) % 8
    return min(d, 8 - d)


def shape_ends(shape: str, start: int) -> list[int]:
    """End buttons a slide shape can reach from start."""
    ok = []
    for e in range(1, 9):
        d = ring(start, e)
        if shape == "-" and d < 2:
            continue
        if shape == "^" and d in (0, 4):
            continue
        if shape in ("s", "z", "w") and d != 4:
            continue
        if shape == "v" and d == 0:
            continue
        ok.append(e)
    return ok


class Grammar:
    """Feed tokens one at a time with push(); allowed() gives a boolean mask
    over the vocabulary for the next token."""

    def __init__(self):
        self.beat_no = -1
        self.pos: Fraction | None = None    # position inside the current beat
        self.used: set[str] = set()         # buttons / sensors used at this moment
        self.state = "top"
        self.flags: list[str] = []          # flags still allowed (in order)
        self.after_len = "top"              # where a finished length goes
        self.len_stage = 0                  # 0: need L, 1: L seen (more L / F / T / done)
        self.slide_from = 0                 # current slide segment's start button
        self.shape = ""
        self.seg_has_len = False
        self.done = False

    # ------------------------------------------------------------ helpers
    def _top(self) -> list[str]:
        out = ["BEAT", "EOS"]
        p = self.pos if self.pos is not None else Fraction(-1)
        out += [f"P{k}" for k in range(GRID) if Fraction(k, GRID) > p]
        out += [f"T{d}" for d in TUPLETS if any(Fraction(n, d) > p for n in range(1, d))]
        if self.pos is not None and self.beat_no >= 0:
            if any(b not in self.used for b in BUTTONS):
                out += ["TAP", "HOLD", "SLIDE"]
            if any(x not in self.used for x in SENS):
                out += ["TOUCH", "THOLD"]
            out.append("PSEUDO")
        return out

    def allowed(self) -> np.ndarray:
        m = np.zeros(V, dtype=bool)
        m[_ids(self._names())] = True
        return m

    def _names(self) -> list[str]:
        s = self.state
        if self.done:
            return ["PAD"]
        if s == "top":
            return self._top()
        if s == "pos_tuplet":                       # T d seen: which n
            p = self.pos_floor
            return [f"N{n}" for n in range(1, self.tuplet_d) if Fraction(n, self.tuplet_d) > p]
        if s == "button":                           # TAP / HOLD / SLIDE: which button
            return [b for b in BUTTONS if b not in self.used]
        if s == "sensor":
            return [x for x in SENS if x not in self.used]
        if s == "flags":
            return self.flags + (self._next_after_flags())
        if s == "len_tuplet":
            return [f"N{n}" for n in range(1, self.tuplet_d)]
        if s == "len":
            if self.len_stage == 0:
                return L_TOK
            return L_TOK + F_TOK + T_TOK + self._after_len_names()
        if s == "path":
            return ["PATH"]
        if s == "path_opts":                        # after PATH: WAIT / PBRK / first segment
            return (["WAIT"] if not (self.path_waited or self.path_brk) else []) + \
                (["PBRK"] if not self.path_brk else []) + [f"SH_{x}" for x in SHAPES]
        if s == "v_mid":
            return [f"B{b}" for b in range(1, 9) if ring(self.slide_from, b) == 2]
        if s == "seg_end":
            if self.shape == "V":
                return [f"B{b}" for b in range(1, 9) if b != self.v_mid and ring(self.v_mid, b) >= 1]
            return [f"B{b}" for b in shape_ends(self.shape, self.slide_from)]
        if s == "seg_after":                        # segment end seen: length, or the next segment
            return L_TOK + [f"SH_{x}" for x in SHAPES]
        if s == "slide_more":                       # path finished: another segment, path, or done
            return [f"SH_{x}" for x in SHAPES] + ["PATH"] + self._top()
        raise RuntimeError(f"bad state {s}")

    def _next_after_flags(self) -> list[str]:
        k = self.kind
        if k in ("TAP", "TOUCH"):
            return self._top()
        if k in ("HOLD", "THOLD"):
            return L_TOK
        if k == "SLIDE":
            return ["PATH"]
        return []

    def _after_len_names(self) -> list[str]:
        if self.after_len == "top":
            return self._top()
        if self.after_len == "wait":                # WAIT length done: PBRK / segment
            return (["PBRK"] if not self.path_brk else []) + [f"SH_{x}" for x in SHAPES]
        if self.after_len == "seg":                 # segment length done
            return [f"SH_{x}" for x in SHAPES] + ["PATH"] + self._top()
        return []

    # ------------------------------------------------------------ push
    def push(self, tok: int) -> None:
        t = VOCAB[tok]
        s = self.state
        if s in ("flags",) and t in self.flags:
            self.flags = self.flags[self.flags.index(t) + 1:]
            if t in ("STAR1", "STAR2"):
                self.flags = []
            if t in ("NOHEAD", "SUDDEN", "HEADTAP", "HEADPAD"):
                self.flags = []
            return
        if s == "flags":                            # flags over: fall through to what comes next
            k = self.kind
            if k in ("HOLD", "THOLD"):
                self.state, self.len_stage, self.after_len = "len", 0, "top"
                return self.push(tok)
            if k == "SLIDE":
                self.state = "path"
                return self.push(tok)
            self.state = "top"
            return self.push(tok)
        if s == "len":
            if t.startswith("L") and t[1:].isdigit():
                self.len_stage = 1
                return
            if t.startswith("F") and t[1:].isdigit():
                return self._len_done()
            if t.startswith("T") and t[1:].isdigit():
                self.tuplet_d = int(t[1:]); self.state = "len_tuplet"
                return
            self._len_done(); return self.push(tok)
        if s == "len_tuplet":
            return self._len_done()
        if s == "pos_tuplet":
            self.pos = Fraction(int(t[1:]), self.tuplet_d)
            self.used, self.state = set(), "top"
            return
        if s == "button":
            if self.kind == "SLIDE":
                self.slide_from = self.slide_start = int(t[1:])
                self.flags = ["BRK", "EX", "NOHEAD", "SUDDEN", "HEADTAP", "HEADPAD"]
            elif self.kind == "TAP":
                self.flags = ["BRK", "EX", "STAR1", "STAR2"]
            else:
                self.flags = ["BRK", "EX"]
            self.used.add(t)
            self.state = "flags"
            return
        if s == "sensor":
            self.used.add(t)
            self.flags, self.state = ["FW", "EX"], "flags"
            return
        if s == "path":                             # t == PATH
            self.slide_from, self.path_waited, self.path_brk = self.slide_start, False, False
            self.state = "path_opts"
            return
        if s == "path_opts":
            if t == "WAIT":
                self.path_waited = True
                self.state, self.len_stage, self.after_len = "len", 0, "wait"
                return
            if t == "PBRK":
                self.path_brk = True
                return
            return self._segment(t)
        if s == "v_mid":
            self.v_mid = int(t[1:]); self.state = "seg_end"
            return
        if s == "seg_end":
            self.slide_from = int(t[1:]); self.state = "seg_after"
            return
        if s == "seg_after":
            if t.startswith("L"):
                self.state, self.len_stage, self.after_len = "len", 1, "seg"
                return
            return self._segment(t)
        if s == "slide_more":
            if t.startswith("SH_"):
                return self._segment(t)
            if t == "PATH":
                self.state = "path"
                return self.push(tok)
            self.state = "top"
            return self.push(tok)
        # top
        if t == "EOS":
            self.done = True
        elif t == "BEAT":
            self.beat_no += 1; self.pos = None; self.used = set()
        elif t[0] == "P" and t[1:].isdigit():
            self.pos = Fraction(int(t[1:]), GRID); self.used = set()
        elif t[0] == "T" and t[1:].isdigit():
            self.tuplet_d = int(t[1:])
            self.pos_floor = self.pos if self.pos is not None else Fraction(-1)
            self.state = "pos_tuplet"
        elif t == "PSEUDO":
            self.used = set()
        elif t in ("TAP", "HOLD", "SLIDE"):
            self.kind, self.state = t, "button"
        elif t in ("TOUCH", "THOLD"):
            self.kind, self.state = t, "sensor"

    def _segment(self, t: str) -> None:
        self.shape = t[3:]
        self.state = "v_mid" if self.shape == "V" else "seg_end"

    def _len_done(self) -> None:
        a = self.after_len
        if a == "top":
            self.state = "top"
        elif a == "wait":
            self.state = "path_opts"
        elif a == "seg":
            self.state = "slide_more"
