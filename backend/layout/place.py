"""Where notes go (layout step 2): buttons for every moment from the rhythm step.

  * Every choice is among the moves the two hands can make (hands.py).
  * Among those, moves are picked the way real charts at the same difficulty
    move: the flow table (how one move follows the previous one: stairs keep
    going, trills alternate ...), plus a little randomness so it isn't one
    fixed pattern. EACH pairs favour mirrored pairs (1/8, 2/7 ...).
  * Notes under a hold or slide go to the other hand, off the slide's zone
    (its buttons and their neighbours, as the checker keeps free) and off a
    released hold's button for a moment after it ends. Before a hold or slide
    is committed, the notes under it are tried on a copy of the hands; if the
    free hand can't play them all, it becomes a tap.
  * Phrases (4 bars) that repeat an earlier phrase's rhythm reuse its buttons,
    mirrored: a repeated musical phrase reads as a repeated pattern, which is
    a clear frame for the creator to build on. If the mirrored copy doesn't
    fit the hands at some note, it continues fresh from there.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from backend.chartcodec.notes import ONE, Hold, Path, Segment, Slide, Tap
from backend.chartcodec.write import _symbol
from backend.checker.playability import ONE_HAND_SEC, slide_clear_beats, slide_zone
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


def _slide_options(b: int, alone: bool) -> list[tuple[str, int, float]]:
    """(shape, end, weight) for a short slide from b: arcs that stay on the
    hand's side, or (only when nothing is played under it) a straight line
    across the middle."""
    own = SIDE[side_of(b)]
    out = []
    for shape, sgn in (("cw", 1), ("ccw", -1)):
        for k in (2, 3):
            e = (b - 1 + sgn * k) % 8 + 1
            if e in own:
                out.append((shape, e, 2.0))
    if alone:
        for k in (3, 4, 5):
            e = (b - 1 + k) % 8 + 1
            out.append(("-", e, 1.0 if k == 4 else 0.5))
    return out


def zone_of(start: int, shape: str, end: int) -> set[int]:
    return slide_zone(start, _symbol(shape, start), end)


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
        self.clear_s = slide_clear_beats(bpm) * self.beat_s
        self.zones: list[tuple[float, set[int]]] = []          # (until, buttons kept free)
        self.hand_of: dict[int, str] = {}                      # moment index -> hand of its hold/slide
        self.moments: list[Moment] = []

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

    def _blocked(self, t: float) -> set[int]:
        return set().union(*[z for until, z in self.zones if until > t + 1e-9]) if self.zones else set()

    def _force(self, m: Moment, t: float) -> str | None:
        if m.under is not None and m.under in self.hand_of:
            return OTHER[self.hand_of[m.under]]
        return forced_hand(t, self.prev_t, self.prev_hand)

    def options(self, m: Moment, idx: int | None = None) -> list[tuple[float, Placed]]:
        """Every playable placement of m with its score (higher = better)."""
        t = self.t(m)
        blocked = self._blocked(t)
        out = []
        if m.each:
            R, L = self.hands["R"], self.hands["L"]
            for r in SIDE["R"]:
                if r in blocked or not R.can(r, t):
                    continue
                for l in SIDE["L"]:
                    if l in blocked or not L.can(l, t):
                        continue
                    s = MIRROR_BONUS * (r + l == 9) + 0.3 * (ring_dist(r, l) == 1)
                    if self.prev_each:
                        pr, plf = self.prev_each
                        move = ring_dist(pr, r) + ring_dist(plf, l)
                        s += EACH_FLOW * (move in (1, 2)) - 0.3 * (move == 0) - 0.4 * max(0, move - 2)
                    out.append((s, Placed("RL", (r, l))))
            return out
        force = self._force(m, t)
        alone = idx is None or not any(x.under == idx for x in self.moments[idx + 1:idx + 64])
        for h in ([force] if force else ["R", "L"]):
            hand = self.hands[h]
            for b in SIDE[h]:
                if b in blocked or not hand.can(b, t):
                    continue
                s = self._flow_score(b)
                if self.prev_single == b:
                    s -= JACK_PENALTY
                if not force:                                  # free choice: favour the hand that rested longer
                    s += 0.2 * (hand.last < self.hands[OTHER[h]].last)
                if m.slide:
                    for shape, e, w in _slide_options(b, alone):
                        out.append((s + math.log(w), Placed(h, (b,), e, shape)))
                else:
                    out.append((s, Placed(h, (b,))))
        return out

    def fits(self, m: Moment, p: Placed, idx: int | None = None) -> bool:
        t = self.t(m)
        if m.each != (p.hand == "RL") or bool(m.slide) != (p.slide_end is not None):
            return False
        if self._blocked(t) & set(p.buttons):
            return False
        if p.hand == "RL":
            return self.hands["R"].can(p.buttons[0], t) and self.hands["L"].can(p.buttons[1], t)
        force = self._force(m, t)
        if not ((force is None or force == p.hand) and self.hands[p.hand].can(p.buttons[0], t)):
            return False
        return not (m.hold or m.slide) or idx is None or self._lookahead(m, p, idx)

    def _lookahead(self, m: Moment, p: Placed, idx: int) -> bool:
        """Can the other hand play every note under this hold/slide?"""
        under = [(j, x) for j, x in enumerate(self.moments[idx + 1:idx + 64], idx + 1) if x.under == idx]
        if not under:
            return True
        trial = copy.copy(self)
        trial.hands = copy.deepcopy(self.hands)
        trial.zones = list(self.zones)
        trial.hand_of = dict(self.hand_of)
        trial.rng = random.Random(0)
        trial.commit(m, p, idx)
        for j, x in under:
            q = trial.choose(x, j, look=False)
            if q is None:
                return False
            trial.commit(x, q, j)
        return True

    def choose(self, m: Moment, idx: int | None = None, look: bool = True) -> Placed | None:
        opts = self.options(m, idx)
        ranked = sorted(opts, key=lambda o: -(o[0] + self._noise()))
        for _, p in ranked:
            if not look or not (m.hold or m.slide) or idx is None or self._lookahead(m, p, idx):
                return p
        return None

    def commit(self, m: Moment, p: Placed, idx: int | None = None) -> None:
        t = self.t(m)
        if p.hand == "RL":
            self.hands["R"].play(p.buttons[0], t, t)
            self.hands["L"].play(p.buttons[1], t, t)
            self.prev_each = p.buttons
            self.prev_single, self.prev_step = None, None
            self.prev_hand = "R" if self.hands["R"].last >= self.hands["L"].last else "L"
        else:
            b = p.buttons[0]
            until, rest, end_b = t, t, b
            if m.hold:
                until = t + float(m.hold) * self.beat_s
                rest = until + ONE_HAND_SEC                               # letting go takes a moment
                self.zones.append((until + self.clear_s, {b}))            # released button rests
            if m.slide:
                until = t + float(1 + m.slide) * self.beat_s
                rest = until + self.clear_s                               # the slide hand rests after
                end_b = p.slide_end
                self.zones.append((until + self.clear_s, zone_of(b, p.slide_shape, p.slide_end)))
            if idx is not None and (m.hold or m.slide):
                self.hand_of[idx] = p.hand
            self.hands[p.hand].play(end_b, t, until, rest)
            if self.prev_single is not None:
                self.prev_step = _step(self.prev_single, b)
            self.prev_single = end_b
            self.prev_hand = p.hand
        self.prev_t = t
        if len(self.zones) > 16:
            self.zones = [z for z in self.zones if z[0] > t]


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
    moments = [copy.copy(m) for m in moments]
    pl = Placer(flow, bpm, first, seed)
    pl.moments = moments
    placed: list[Placed | None] = []
    seen: dict[tuple, list[Placed]] = {}
    reused = dropped = downgraded = 0
    phrases: dict[int, list[int]] = {}
    for i, m in enumerate(moments):
        phrases.setdefault(m.slot // PHRASE_SLOTS, []).append(i)
    for ph, ids in sorted(phrases.items()):
        sig = _signature([moments[i] for i in ids], ph * PHRASE_SLOTS)
        copy_ = [mirrored(p) for p in seen[sig]] if sig in seen else None
        out = []
        for j, i in enumerate(ids):
            m = moments[i]
            p = copy_[j] if copy_ is not None and pl.fits(m, copy_[j], i) else None
            if p is None:
                copy_ = None                                       # continue fresh from here
                p = pl.choose(m, i)
                if p is None and (m.hold or m.slide):              # the free hand can't play what's under it
                    m.hold = m.slide = Fraction(0)
                    for x in moments[i + 1:i + 64]:
                        if x.under == i:
                            x.under = None
                    downgraded += 1
                    p = pl.choose(m, i)
            else:
                reused += 1
            if p is None:
                dropped += 1
                out.append(None)
                continue
            pl.commit(m, p, i)
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
    return notes, {"reused_from_mirrored_phrase": reused, "dropped": dropped,
                   "holds_slides_made_taps": downgraded}
