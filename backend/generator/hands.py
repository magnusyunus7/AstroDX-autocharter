"""Which buttons: a simple two-hand model (v1: playable before interesting).

Buttons 1-8 run clockwise from the top right. The right hand covers 1-4, the
left hand 5-8 (mirror pairs: 1<->8, 2<->7, 3<->6, 4<->5).
  * fast notes alternate hands (no one-hand jacks at speed)
  * each hand moves step by step around its half, so runs flow
  * slow notes may repeat a button or jump, like real charts
  * EACH: a mirrored pair, sometimes an opposite pair
"""
from __future__ import annotations

import random

RIGHT = [1, 2, 3, 4]
LEFT = [8, 7, 6, 5]                 # index 0 = top, like RIGHT
MIRROR = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
FAST_SEC = 0.20                     # closer than this: alternate hands
SLOW_SEC = 0.40                     # further than this: free to repeat / jump


def assign(times: list[float], each: list[bool], seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    pos = {"R": 1, "L": 1}          # index within the hand's half (0 = top)
    last_hand, last_t = "L", -1e9
    out: list[list[int]] = []
    for t, is_each in zip(times, each):
        gap = t - last_t
        if is_each:
            for h in ("R", "L"):
                pos[h] = _step(pos[h], rng, gap)
            r = RIGHT[pos["R"]]
            pair = [r, MIRROR[r]] if rng.random() < 0.8 else [r, (r + 3) % 8 + 1]
            if pair[0] == pair[1]:
                pair = [r, MIRROR[r]]
            out.append(sorted(set(pair)))
            last_hand, last_t = "R" if rng.random() < 0.5 else "L", t
            continue
        if gap < FAST_SEC:
            hand = "R" if last_hand == "L" else "L"
        else:
            hand = rng.choice("RL")
        pos[hand] = _step(pos[hand], rng, gap)
        out.append([(RIGHT if hand == "R" else LEFT)[pos[hand]]])
        last_hand, last_t = hand, t
    return out


def _step(i: int, rng: random.Random, gap: float) -> int:
    if gap > SLOW_SEC:
        moves = (-2, -1, 0, 1, 2)
    elif gap < FAST_SEC:
        moves = (-1, 1)             # keep moving at speed: flowing runs, no jacks
    else:
        moves = (-1, 0, 1)
    j = i + rng.choice(moves)
    if j < 0 or j > 3:
        j = i - (j - i) if 0 <= i - (j - i) <= 3 else i
    return max(0, min(3, j))