"""Two simulated hands (layout step 2's rules).

Kept deliberately simple, so every layout is playable without thinking:
  * the right hand plays buttons 1-4, the left hand 5-8: hands never cross;
  * an EACH is one note per hand;
  * notes closer than ONE_HAND_SEC alternate hands (one hand can't play them
    both); slower runs may stay on one hand, so stairs (1,2,3,4) are possible;
  * a hand moves at most as far as the time since its own last note allows
    (REACH), and repeats a button (jack) only with ONE_HAND_SEC between;
  * a hand holding a note or playing a slide is busy until it ends (the
    rhythm step leaves the gap after it empty, so the other hand never has
    to play alone).
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.checker.playability import ONE_HAND_SEC

SIDE = {"R": (1, 2, 3, 4), "L": (5, 6, 7, 8)}
OTHER = {"R": "L", "L": "R"}
# (seconds since this hand's last note, farthest move around the ring)
REACH = ((0.10, 1), (0.16, 2), (0.24, 3), (0.30, 4))


def side_of(b: int) -> str:
    return "R" if b <= 4 else "L"


def ring_dist(a: int, b: int) -> int:
    d = abs(a - b) % 8
    return min(d, 8 - d)


def max_move(dt: float) -> int:
    best = 0
    for sec, d in REACH:
        if dt >= sec - 1e-9:
            best = d
    return best


def mirror(b: int) -> int:
    """Left-right mirror of the ring (1<->8, 2<->7, 3<->6, 4<->5)."""
    return 9 - b


@dataclass
class Hand:
    name: str
    button: int | None = None
    last: float = -1e9            # time of its last note
    busy_until: float = -1e9      # holding / sliding until

    def can(self, b: int, t: float) -> bool:
        if b not in SIDE[self.name] or t < self.busy_until - 1e-9:
            return False
        if self.button is None:
            return True
        dt = t - self.last
        d = ring_dist(self.button, b)
        if d == 0:
            return dt >= ONE_HAND_SEC - 1e-9
        return d <= max_move(dt)

    def play(self, b_end: int, t: float, until: float, rest: float | None = None) -> None:
        """A note at t on b; the hand lets go at `until` (hold / slide end; t
        for a tap) and is free again at `rest` (default: until)."""
        self.button, self.last = b_end, max(t, until)
        self.busy_until = max(self.last, rest if rest is not None else until)


def forced_hand(t: float, prev_t: float | None, prev_hand: str | None) -> str | None:
    """Notes closer than ONE_HAND_SEC alternate hands."""
    if prev_t is not None and prev_hand is not None and t - prev_t < ONE_HAND_SEC - 1e-9:
        return OTHER[prev_hand]
    return None
