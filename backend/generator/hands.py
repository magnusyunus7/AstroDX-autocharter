"""Which buttons: a two-hand model (playable before interesting).

Buttons 1-8 run clockwise from the top right. The right hand covers 1-4, the
left hand 5-8 (mirror pairs: 1<->8, 2<->7, 3<->6, 4<->5).
  * fast notes alternate hands (no one-hand jacks at speed)
  * each hand moves step by step around its half, so runs flow
  * slow notes may repeat a button or jump, like real charts
  * EACH: a mirrored pair, sometimes an opposite pair
  * a hold or slide keeps its hand busy; meanwhile the other hand plays alone
    (kinds.py made sure that's one-hand speed). A slide with notes during it
    stays in its own half; a free slide may cross the screen
  * touch notes go on the sensor next to where the hand would have tapped
  * from a slide's star until its finish room ends (kinds: clear_until),
    nothing lands on or next to its buttons, except the slide hand's next note,
    which goes right next to the slide's end (the shortest move); a held button
    gets the same room after release. If a note has no safe button on either
    side, it's left out
"""
from __future__ import annotations

import random
from fractions import Fraction
from dataclasses import dataclass

from backend.checker.playability import next_spots, slide_next_beats, slide_zone
from backend.models.chart import ring_distance

RIGHT = [1, 2, 3, 4]
LEFT = [8, 7, 6, 5]                 # index 0 = top, like RIGHT
MIRROR = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
FAST_SEC = 0.20                     # closer than this: alternate hands
ONE_HAND_SEC = 0.15                 # one hand twice in a row: at least this far apart
SLOW_SEC = 0.40                     # further than this: free to repeat / jump
CW_IS_RIGHT = {1, 2, 7, 8}          # simai: from these, '>' turns clockwise; from 3-6, '<' does
MIRROR_SHAPE = {">": "<", "<": ">", "p": "q", "q": "p", "-": "-", "v": "v", "w": "w"}

# slide shape weights; arcs by how many buttons they pass (charts mostly use 3)
SHAPE_WEIGHT = {"-": 0.55, "arc": 0.25, "v": 0.10, "p": 0.04, "w": 0.04}
ARC_STEPS = {1: 0.1, 2: 0.4, 3: 1.0, 4: 0.5, 5: 0.3, 6: 0.2, 7: 0.3}


@dataclass
class Placed:
    kind: str                       # tap / hold / slide / touch
    button: int = 0                 # tap/hold/slide start
    shape: str = ""                 # slide
    end: int = 0                    # slide end button
    sensor: str = ""                # touch


def _arc(start: int, end: int, clockwise: bool) -> str:
    cw_symbol = ">" if start in CW_IS_RIGHT else "<"
    return cw_symbol if clockwise else ("<" if cw_symbol == ">" else ">")


def _cw_steps(a: int, b: int) -> int:
    return (b - a) % 8


def slide_options(start: int, half: list[int], restricted: bool, long: bool,
                  allow_wifi: bool) -> list[tuple[float, str, int]]:
    """(weight, shape, end) for slides from `start`. Every shape works at [8:1]
    (charts do 3-button arcs and full straights that fast); the long ones
    (loops, wifi, 5+ button arcs) need a longer slide."""
    opts = []
    for end in range(1, 9):
        if end == start:
            continue
        if restricted and end not in half:
            continue
        d = ring_distance(start, end)
        if d >= 2:
            opts.append((SHAPE_WEIGHT["-"] / 3, "-", end))
        if 1 <= d <= 3:
            opts.append((SHAPE_WEIGHT["v"] / 3, "v", end))
        for cw in (True, False):
            k = _cw_steps(start, end) if cw else _cw_steps(end, start)
            path = [(start - 1 + (s if cw else -s)) % 8 + 1 for s in range(1, k + 1)]
            if restricted and any(b not in half for b in path):
                continue
            if k <= 4 or long:
                opts.append((SHAPE_WEIGHT["arc"] * ARC_STEPS[k] / 4, _arc(start, end, cw), end))
        if not restricted and d >= 2 and long:
            opts.append((SHAPE_WEIGHT["p"] / 4, "p" if (start + end) % 2 else "q", end))
        if not restricted and allow_wifi and d == 4 and long:
            opts.append((SHAPE_WEIGHT["w"], "w", end))
    return opts


def _pick(opts, rng):
    total = sum(w for w, *_ in opts)
    r = rng.random() * total
    for w, *rest in opts:
        r -= w
        if r <= 0:
            return rest
    return opts[-1][1:]


def _touch_sensor(button: int, strong: bool, rng: random.Random) -> str:
    r = rng.random()
    if strong and r < 0.25:
        return "C"
    return ("B" if r < 0.6 else "E") + str(button)


def assign(times: list[float], each: list[bool], seed: int = 0, kinds=None,
           beat: float = 0.5, allow_wifi: bool = False, strong=None,
           clear_sec: float = 0.0) -> list[list[Placed]]:
    """One list of Placed per moment (empty: no safe button, leave it out).
    `kinds` (from kinds.decide) adds holds, slides and touch; without it every
    note is a tap. clear_sec: room a held button gets after the hold ends."""
    rng = random.Random(seed)
    next_sec = slide_next_beats(60.0 / beat) * beat   # slide hand's soonest next note
    pos = {"R": 1, "L": 1}          # index within the hand's half (0 = top)
    busy = {"R": -1e9, "L": -1e9}   # hand busy until this time
    forbid: dict[int, list] = {}    # button -> [(until, slide id or None)]: keep it free until then
    cont: dict[str, tuple | None] = {"R": None, "L": None}   # hand's slide: (until, next spots, id)
    hand_t = {"R": -1e9, "L": -1e9}  # each hand's last note
    busy_btn: dict[str, int] = {}    # button a busy hand is on
    zone_id = [0]
    last_hand, last_t = "L", -1e9
    out: list[list[Placed]] = []

    def halves(h):
        return RIGHT if h == "R" else LEFT

    def block(start, shape, end, until, hand=None, ready=0.0) -> None:
        zone_id[0] += 1
        for b in slide_zone(start, shape, end):
            forbid.setdefault(b, []).append((until, zone_id[0]))
        if hand:
            cont[hand] = (until, next_spots(start, shape, end), zone_id[0], ready)

    def ok(b: int, t: float, exempt=None) -> bool:
        return all(until <= t + 1e-6 or (exempt is not None and zid == exempt) for until, zid in forbid.get(b, ()))

    def allowed_index(h: str, i: int, t: float):
        """Where hand h may play at t: i, or the nearest free spot in its half.
        Right after its own slide (until the finish room ends) only right next
        to the slide's end: the shortest move, the one tradeoff for little
        time. None if there's no such spot."""
        c = cont[h]
        if c and t >= c[0] - 1e-6:
            c = cont[h] = None
        if c:
            until, spots, zid, ready = c
            if t < ready - 1e-6:
                return None
            near = [j for j in range(4) if halves(h)[j] in spots and ok(halves(h)[j], t, exempt=zid)]
            return min(near, key=lambda j: (abs(j - i), j)) if near else None
        for j in sorted(range(4), key=lambda j: (abs(j - i), j)):
            if ok(halves(h)[j], t):
                return j
        return None

    def used(h: str) -> None:
        cont[h] = None                                   # only the next note is tied to the slide end

    def note_for(h: str, kind: str, k, t: float, button: int) -> Placed:
        if kind == "slide":
            long = k.length >= 2 * (Fraction(1, 4) if 60.0 / beat < 120 else Fraction(1, 2))
            opts = slide_options(button, halves(h), k.restricted, long, allow_wifi)
            if not opts:
                return Placed("tap", button)
            shape, end = _pick(opts, rng)
            block(button, shape, end, k.clear_until, hand=h, ready=k.busy_until + next_sec)
            # the hand ends where the slide ends (nearest spot in its own half)
            pos[h] = min(range(4), key=lambda i: ring_distance(halves(h)[i], end))
            return Placed("slide", button, shape=shape, end=end)
        if kind == "hold":                               # its button gets room after release
            forbid.setdefault(button, []).append((k.busy_until + clear_sec, None))
            return Placed("hold", button)
        if kind == "touch":
            return Placed("touch", sensor=_touch_sensor(button, bool(strong and strong(t)), rng))
        return Placed("tap", button)

    for i, (t, is_each) in enumerate(zip(times, each)):
        k = kinds[i] if kinds else None
        kind = k.kind if k else "tap"
        gap = t - last_t
        free = [h for h in ("R", "L") if busy[h] <= t + 1e-6]
        if is_each and len(free) == 2:
            steps = {hh: allowed_index(hh, _step(pos[hh], rng, gap), t) for hh in ("R", "L")}
            if steps["R"] is not None and steps["L"] is not None:
                pos.update(steps)
                used("R"); used("L")
                r = RIGHT[pos["R"]]
                # mirrored pair, sometimes the opposite button (1/5, 2/6, ...); both left = 5-8
                left = MIRROR[r] if (rng.random() < 0.8 or kind != "tap") else (r + 3) % 8 + 1
                if not ok(left, t):
                    left = LEFT[pos["L"]]
                btn = {"R": r, "L": left}
                main = rng.choice("RL")
                other = "L" if main == "R" else "R"
                notes = [note_for(main, kind, k, t, btn[main])]
                mirror_ok = notes[0].kind != "slide" or ok(MIRROR[notes[0].button], t)
                if k and k.partner == kind and kind in ("hold", "slide") and notes[0].kind == kind and mirror_ok:
                    if kind == "slide":                      # mirrored double slide
                        m = notes[0]
                        notes.append(Placed("slide", MIRROR[m.button], MIRROR_SHAPE[m.shape], MIRROR[m.end]))
                        block(MIRROR[m.button], MIRROR_SHAPE[m.shape], MIRROR[m.end], k.clear_until,
                              hand=other, ready=k.busy_until + next_sec)
                        pos[other] = min(range(4), key=lambda j: ring_distance(halves(other)[j], MIRROR[m.end]))
                    else:
                        notes.append(note_for(other, "hold", k, t, btn[other]))
                    busy[other] = k.busy_until
                else:
                    notes.append(Placed("tap", btn[other]))
                if kind in ("hold", "slide"):
                    busy[main] = k.busy_until
                out.append(notes)
                last_hand, last_t = main if kind == "tap" else other, t
                hand_t["R"] = hand_t["L"] = t
                continue
            # no room for both: play it as a single note
        if len(free) == 1:
            order = free
        elif not free:                                   # shouldn't happen; least busy hand
            order = [min(busy, key=busy.get)]
        elif gap < FAST_SEC:
            order = ["R" if last_hand == "L" else "L"]
            order.append("L" if order[0] == "R" else "R")
        else:
            order = rng.sample(["R", "L"], 2)
        placed_note = None
        for hand in order:
            if hand != order[0] and t - hand_t[hand] < ONE_HAND_SEC:
                continue                                 # this hand just played: too fast to go again
            c = cont[hand]
            if c and t < c[0] - 1e-6:
                # right after its own slide: not before `ready` ({8} after the end),
                # then only right next to the slide's end (maybe just over the middle)
                if t < c[3] - 1e-6:
                    continue
                spots = [b for b in sorted(c[1]) if ok(b, t, exempt=c[2])
                         and not any(busy_btn.get(o) == b and busy[o] > t + 1e-6
                                     for o in ("R", "L") if o != hand)]
                if not spots:
                    continue
                b = min(spots, key=lambda b: ring_distance(b, halves(hand)[pos[hand]]))
                pos[hand] = min(range(4), key=lambda j: ring_distance(halves(hand)[j], b))
            else:
                j = allowed_index(hand, _step(pos[hand], rng, gap if len(free) == 2 else max(gap, FAST_SEC)), t)
                if j is None:
                    continue                             # this hand's side is blocked: try the other
                pos[hand] = j
                b = halves(hand)[j]
            used(hand)
            placed_note = note_for(hand, kind, k, t, b)
            break
        if placed_note is None:                          # nowhere safe: leave this note out
            out.append([])
            continue
        out.append([placed_note])
        if kind in ("hold", "slide"):
            busy[hand] = k.busy_until
            busy_btn[hand] = placed_note.button
        last_hand, last_t = hand, t
        hand_t[hand] = t
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