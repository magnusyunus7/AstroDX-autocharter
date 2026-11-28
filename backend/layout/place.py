"""Where notes go (layout step 2): buttons for every moment from the rhythm step.

  * Every choice is among the moves the two hands can make (hands.py).
  * Among those, moves are picked the way real charts at the same difficulty
    move: the flow table (how one move follows the previous one: stairs keep
    going, trills alternate ...), plus a little randomness so it isn't one
    fixed pattern. EACH pairs favour mirrored pairs (1/8, 2/7 ...).
  * Phrases (4 bars) that repeat an earlier phrase's rhythm reuse its buttons,
    mirrored: a repeated musical phrase reads as a repeated pattern, which is
    a clear frame for the creator to build on. If the mirrored copy doesn't
    fit the hands at some note, it continues fresh from there.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from backend.chartcodec.notes import ONE, Hold, Path, Segment, Slide, Tap
from backend.layout.hands import OTHER, SIDE, Hand, forced_hand, mirror, ring_dist, side_of
from backend.layout.rhythm import Moment

PHRASE_SLOTS = 4 * 16
NOISE = 0.6                       # randomness of choices (Gumbel scale)
JACK_PENALTY = 1.5                # same button again (single notes)
MIRROR_BONUS = 1.0                # EACH on mirrored buttons (1/8, 2/7, ...)
EACH_FLOW = 1.2                   # EACH pair moving on smoothly from the previous one


@dataclass
class Placed:
    hand: str                     # "R" / "L" / "RL" (EACH)
    buttons: tuple[int, ...]      # one per hand (EACH: right, left)
    slide_end: int | None = None
    slide_shape: str | None = None


def _step(a: int, b: int) -> int:
    return ((b - a + 3) % 8) - 3


def _slide_options(b: int) -> list[tuple[str, int, float]]:
    """(shape, end, weight) for a short slide from b: arcs that stay on the
    hand's side, or a straight line across the middle."""
    own = SIDE[side_of(b)]
    out = []
    for shape, sgn in (("cw", 1), ("ccw", -1)):
        for k in (2, 3):
            e = (b - 1 + sgn * k) % 8 + 1
            if e in own:
                out.append((shape, e, 2.0))
    for k in (3, 4, 5):
        e = (b - 1 + k) % 8 + 1
        out.append(("-", e, 1.0 if k == 4 else 0.5))
    return out


class Placer:
    def __init__(self, flow: np.ndarray, bpm: float, first: float, seed: int = 0):
        self.flow = np.asarray(flow)
        self.first_move = np.log(np.exp(self.flow).mean(0))      # a move with no move before it
        self.beat_s = 60.0 / bpm
        self.first = first
        self.rng = random.Random(seed)
        self.hands = {"R": Hand("R"), "L": Hand("L")}
        self.prev_t: float | None = None
        self.prev_hand: str | None = None
        self.prev_single: int | None = None
        self.prev_step: int | None = None
        self.prev_each: tuple[int, int] | None = None

    def t(self, m: Moment) -> float:
        return self.first + m.slot * self.beat_s / 4

    def _noise(self) -> float:
        u = max(1e-12, self.rng.random())
        return -NOISE * math.log(-math.log(u))

    def _flow_score(self, b: int) -> float:
        if self.prev_single is None:
            return 0.0
        st = _step(self.prev_single, b)
        if self.prev_step is None:
            return float(self.first_move[st + 3])
        return float(self.flow[self.prev_step + 3, st + 3])

    # ---------------------------------------------------------- one moment

    def options(self, m: Moment) -> list[tuple[float, Placed]]:
        """Every playable placement of m with its score (higher = better)."""
        t = self.t(m)
        out = []
        if m.each:
            R, L = self.hands["R"], self.hands["L"]
            for r in SIDE["R"]:
                if not R.can(r, t):
                    continue
                for l in SIDE["L"]:
                    if not L.can(l, t):
                        continue
                    s = MIRROR_BONUS * (r + l == 9) + 0.3 * (ring_dist(r, l) == 1)
                    if self.prev_each:
                        pr, plf = self.prev_each
                        move = ring_dist(pr, r) + ring_dist(plf, l)
                        s += EACH_FLOW * (move in (1, 2)) - 0.3 * (move == 0) - 0.4 * max(0, move - 2)
                    out.append((s, Placed("RL", (r, l))))
            return out
        force = forced_hand(t, self.prev_t, self.prev_hand)
        for h in ([force] if force else ["R", "L"]):
            hand = self.hands[h]
            for b in SIDE[h]:
                if not hand.can(b, t):
                    continue
                s = self._flow_score(b)
                if self.prev_single == b:
                    s -= JACK_PENALTY
                if not force:                                  # free choice: favour the hand that rested longer
                    s += 0.2 * (hand.last < self.hands[OTHER[h]].last)
                if m.slide:
                    for shape, e, w in _slide_options(b):
                        out.append((s + math.log(w), Placed(h, (b,), e, shape)))
                else:
                    out.append((s, Placed(h, (b,))))
        return out

    def fits(self, m: Moment, p: Placed) -> bool:
        t = self.t(m)
        if m.each != (p.hand == "RL") or bool(m.slide) != (p.slide_end is not None):
            return False
        if p.hand == "RL":
            return self.hands["R"].can(p.buttons[0], t) and self.hands["L"].can(p.buttons[1], t)
        force = forced_hand(t, self.prev_t, self.prev_hand)
        return (force is None or force == p.hand) and self.hands[p.hand].can(p.buttons[0], t)

    def choose(self, m: Moment) -> Placed | None:
        opts = self.options(m)
        if not opts:
            return None
        return max(opts, key=lambda o: o[0] + self._noise())[1]

    def commit(self, m: Moment, p: Placed) -> None:
        t = self.t(m)
        if p.hand == "RL":
            self.hands["R"].play(p.buttons[0], t, t)
            self.hands["L"].play(p.buttons[1], t, t)
            self.prev_each = p.buttons
            self.prev_single, self.prev_step = None, None
            self.prev_hand = "R" if self.hands["R"].last >= self.hands["L"].last else "L"
        else:
            b = p.buttons[0]
            until, end_b = t, b
            if m.hold:
                until = t + float(m.hold) * self.beat_s
            if m.slide:
                until = t + float(1 + m.slide) * self.beat_s
                end_b = p.slide_end
            self.hands[p.hand].play(end_b, t, until)
            if self.prev_single is not None:
                self.prev_step = _step(self.prev_single, b)
            self.prev_single = end_b
            self.prev_hand = p.hand
        self.prev_t = t


def mirrored(p: Placed) -> Placed:
    if p.hand == "RL":
        return Placed("RL", (mirror(p.buttons[1]), mirror(p.buttons[0])))
    shape = {"cw": "ccw", "ccw": "cw"}.get(p.slide_shape, p.slide_shape)
    return Placed(OTHER[p.hand], (mirror(p.buttons[0]),),
                  mirror(p.slide_end) if p.slide_end else None, shape)


def _signature(ms: list[Moment], start: int) -> tuple:
    return tuple((m.slot - start, m.each, m.hold, m.slide) for m in ms)


def place(moments: list[Moment], flow, bpm: float, first: float, seed: int = 0) -> tuple[list, dict]:
    """Notes (chart codec objects) for the moments, and some counts."""
    pl = Placer(flow, bpm, first, seed)
    placed: list[Placed | None] = []
    seen: dict[tuple, list[Placed]] = {}
    reused = dropped = 0
    phrases: dict[int, list[int]] = {}
    for i, m in enumerate(moments):
        phrases.setdefault(m.slot // PHRASE_SLOTS, []).append(i)
    for ph, ids in sorted(phrases.items()):
        ms = [moments[i] for i in ids]
        sig = _signature(ms, ph * PHRASE_SLOTS)
        copy = [mirrored(p) for p in seen[sig]] if sig in seen else None
        out = []
        for j, m in enumerate(ms):
            p = copy[j] if copy is not None and pl.fits(m, copy[j]) else None
            if p is None:
                copy = None                                       # continue fresh from here
                p = pl.choose(m)
            else:
                reused += 1
            if p is None:
                dropped += 1
                out.append(None)
                continue
            pl.commit(m, p)
            out.append(p)
        if all(p is not None for p in out):
            seen[sig] = out
        placed += out
    notes = []
    for m, p in zip(moments, placed):
        if p is None:
            continue
        beat = m.beat
        if p.hand == "RL":
            notes += [Tap(beat, p.buttons[0]), Tap(beat, p.buttons[1])]
        elif m.slide:
            seg = Segment(p.slide_shape, p.slide_end, None, Fraction(m.slide))
            notes.append(Slide(beat, p.buttons[0], (Path((seg,), ONE),)))
        elif m.hold:
            notes.append(Hold(beat, p.buttons[0], Fraction(m.hold)))
        else:
            notes.append(Tap(beat, p.buttons[0]))
    return notes, {"reused_from_mirrored_phrase": reused, "dropped": dropped}
