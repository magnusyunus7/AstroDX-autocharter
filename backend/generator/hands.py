"""Which buttons: two hands travelling around the ring, in patterns.

Buttons 1-8 run clockwise from the top right. Both hands can go anywhere; at
rest the right hand sits on 1-4 and the left on 5-8.
  * fast notes alternate hands and mostly move to a nearby button, like real
    charts (fast runs in the sample charts: ~53% to a neighbouring button).
    Runs follow a pattern for a few notes: spin (one direction), wiggle
    (back and forth next door), stairs (2 at a time), trill (two buttons),
    or free; the step mix matches the sample charts (STEP_PRIOR)
  * a hand can only move so far in the time since its own last note
  * slow notes go to the hand on that side; they may repeat or jump
  * EACH: two buttons 1-4 apart (mirror, neighbours, opposite), near the hands
  * a hold or slide keeps its hand busy; the other hand plays alone and never
    on (or across) the held button. A slide with notes during it stays on its
    side of the screen; a free slide may cross
  * touch notes go on the sensor next to where the hand would have tapped
  * from a slide's star until its finish room ends (kinds: clear_until),
    nothing lands on or next to its buttons, except the slide hand's next note,
    which goes right next to the slide's end, no sooner than a {8} after it; a
    held button gets the same room after release. If a note has no safe
    button for either hand, it's left out
"""
from __future__ import annotations

import math
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
    sub: int = 0                    # burst: slot (0, 1, 2, ...) within the burst


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
            if k < 2:
                continue                             # a curve passes at least 2 buttons (1<7, not 1<8)
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


# step (next button - previous, -4..3) mix by gap between notes, from the sample charts
STEP_PRIOR = {
    "fast": {0: .06, 1: .30, -1: .23, 2: .12, -2: .11, 3: .055, -3: .058, -4: .068},   # < 0.13 s
    "mid": {0: .21, 1: .16, -1: .18, 2: .12, -2: .12, 3: .08, -3: .07, -4: .05},       # < 0.20 s
    "slow": {0: .19, 1: .18, -1: .19, 2: .10, -2: .12, 3: .07, -3: .05, -4: .09},
}
FAST_RUN_SEC = 0.13
# patterns for fast runs: (weight, name)
PATTERNS = ((0.40, "spin"), (0.18, "wiggle"), (0.10, "stairs"), (0.10, "trill"), (0.22, "free"))
PATTERN_LEN = (6, 12)               # notes before picking a new pattern
PATTERN_BOOST = 30.0                # how strongly a run follows its pattern
EACH_DIST = {1: .23, 2: .29, 3: .24, 4: .23}   # ring distance between EACH buttons
# slower notes (not in fast runs) follow simple patterns too: (weight, name)
SLOW_PATTERNS = ((0.35, "alternate"), (0.25, "walk"), (0.10, "repeat"), (0.30, "free"))
SLOW_BOOST = 6.0
# a bar with the same rhythm as a recent one repeats its buttons (or their mirror)
MOTIF_MIN_NOTES = 3
MOTIF_LOOKBACK = 8                  # bars
MOTIF_REUSE = 0.90                  # chance to repeat when a match exists
MOTIF_OVERLAP = 0.6                 # share of note spots the two bars must have in common
MOTIF_MIRROR_NEXT = 0.6             # right after the original: mirrored (call and response)
MOTIF_BOOST = 40.0
# how a repeated bar may change: (name, (weight, button map))
MOTIF_TRANSFORMS = (
    ("same", (0.35, lambda b: b)),
    ("mirror", (0.35, lambda b: 9 - b)),                    # left <-> right
    ("turn+1", (0.08, lambda b: b % 8 + 1)),                # the shape moved one button on
    ("turn-1", (0.08, lambda b: (b - 2) % 8 + 1)),
    ("flip", (0.07, lambda b: 5 - b if b <= 4 else 13 - b)),  # top <-> bottom
    ("turn+2", (0.07, lambda b: (b + 1) % 8 + 1)),
)
# both hands share the work: lean away from the side that had more notes lately
BALANCE_WINDOW = 24
BALANCE_PULL = 16.0
SIDE = {b: ("R" if b <= 4 else "L") for b in range(1, 9)}


def _step(a: int, b: int) -> int:
    """b - a around the ring, as -4..3."""
    return (b - a + 4) % 8 - 4


def _move_limit(dt: float) -> int:
    """How far one hand can go in dt seconds since its own last note."""
    if dt < 0.10:
        return 1
    if dt < 0.15:
        return 2
    if dt < 0.22:
        return 3
    return 4


def _between(a: int, b: int, x: int) -> bool:
    """x lies on the short way from a to b (exclusive of a, inclusive of b)."""
    s = _step(a, b)
    return any((a - 1 + (k if s > 0 else -k)) % 8 + 1 == x for k in range(1, abs(s) + 1))


def assign(times: list[float], each: list[bool], seed: int = 0, kinds=None,
           beat: float = 0.5, allow_wifi: bool = False, strong=None,
           clear_sec: float = 0.0, beats=None) -> list[list[Placed]]:
    """One list of Placed per moment (empty: no safe button, leave it out).
    `kinds` (from kinds.decide) adds holds, slides and touch; without it every
    note is a tap. clear_sec: room a held button gets after the hold ends.
    beats: chart beat of each moment (enables repeating bar patterns)."""
    rng = random.Random(seed)
    next_sec = slide_next_beats(60.0 / beat) * beat   # slide hand's soonest next note
    at = {"R": 2, "L": 7}           # each hand's button
    hand_t = {"R": -1e9, "L": -1e9}  # each hand's last note
    busy = {"R": -1e9, "L": -1e9}   # hand busy (hold / slide) until this time
    held: dict[str, int] = {}       # button a busy hand is on
    forbid: dict[int, list] = {}    # button -> [(until, slide id or None)]: keep it free until then
    cont: dict[str, tuple | None] = {"R": None, "L": None}   # hand's slide: (until, spots, id, ready)
    zone_id = [0]
    last_hand, last_t, last_btn = "L", -1e9, 7
    pattern = {"name": "free", "left": 0, "dir": 1, "d": 1, "n": 0}
    slow_pat = {"name": "free", "left": 0, "dir": 1}
    recent: list[str] = []          # sides of the last notes (keeps both hands busy)
    out: list[list[Placed]] = []

    # ---- bars with the same rhythm repeat a pattern (as is, or mirrored)
    bar_of, sig = [], {}
    for i in range(len(times)):
        b_ = int(beats[i] // 4) if beats is not None else -1
        bar_of.append(b_)
        sig.setdefault(b_, []).append(beats[i] - 4 * b_ if beats is not None else 0)
    sig = {b_: tuple(v) for b_, v in sig.items()}
    played: dict[int, dict] = {}    # bar -> {spot in bar: buttons}
    want: dict | None = None        # this bar's pattern from an earlier bar

    def other(h):
        return "L" if h == "R" else "R"

    def block(start, shape, end, until, hand=None, ready=0.0) -> None:
        zone_id[0] += 1
        for b in slide_zone(start, shape, end):
            forbid.setdefault(b, []).append((until, zone_id[0]))
        if hand:
            cont[hand] = (until, next_spots(start, shape, end), zone_id[0], ready)

    def ok(b: int, t: float, exempt=None) -> bool:
        return all(until <= t + 1e-6 or (exempt is not None and zid == exempt) for until, zid in forbid.get(b, ()))

    def is_busy(h, t):
        return busy[h] > t + 1e-6

    def reachable(h: str, b: int, t: float) -> bool:
        """Hand h can play b at t: free, close enough, not on/through the other
        hand's held button, and (right after its own slide) only on the spots
        next to the slide's end."""
        if is_busy(h, t):
            return False
        o = other(h)
        if is_busy(o, t) and (held.get(o) == b or _between(at[h], b, held.get(o, 0))):
            return False
        c = cont[h]
        if c and t < c[0] - 1e-6:
            if t < c[3] - 1e-6 or b not in c[1] or not ok(b, t, exempt=c[2]):
                return False
            return True
        if not ok(b, t):
            return False
        return ring_distance(at[h], b) <= _move_limit(t - hand_t[h])

    def new_pattern():
        name = _pick([(w, n) for w, n in PATTERNS], rng)[0]
        pattern.update(name=name, left=rng.randint(*PATTERN_LEN), dir=rng.choice((1, -1)),
                       d=rng.choice((2, 3, 4)), n=0)

    def pattern_step() -> int | None:
        p = pattern
        if p["name"] == "spin":
            return p["dir"]
        if p["name"] == "stairs":
            return 2 * p["dir"]
        if p["name"] == "wiggle":
            return p["dir"] * (1 if p["n"] % 2 == 0 else -1)
        if p["name"] == "trill":                         # two buttons d apart, back and forth
            s = p["d"] * p["dir"] * (1 if p["n"] % 2 == 0 else -1)
            return (s + 4) % 8 - 4
        return None

    def weights(prev: int, gap: float, in_run: bool) -> dict[int, float]:
        cls = "fast" if gap < FAST_RUN_SEC else "mid" if gap < FAST_SEC else "slow"
        prior = STEP_PRIOR[cls]
        want = pattern_step() if in_run else None
        w = {}
        for b in range(1, 9):
            s = _step(prev, b)
            w[b] = prior.get(s, 0.01) * (PATTERN_BOOST if want is not None and s == want else 1.0)
        return w

    def start_bar(b_: int, t: float, first_pos):
        """Repeat the most recent bar with (mostly) the same rhythm: notes on
        the same spots get that bar's buttons, as they were, mirrored, or the
        whole shape moved round the ring - whichever the hands can start from."""
        nonlocal want
        want = None
        if beats is None or b_ < 0 or len(sig[b_]) < MOTIF_MIN_NOTES:
            return
        here = set(sig[b_])
        for back in range(1, MOTIF_LOOKBACK + 1):
            src = played.get(b_ - back)
            if not src:
                continue
            shared = here & set(src)
            if len(shared) < MOTIF_OVERLAP * max(len(here), len(src)):
                continue
            if rng.random() > MOTIF_REUSE:
                return
            first = src.get(first_pos, ((), None))[0]
            options = []
            for name, f in MOTIF_TRANSFORMS:
                w_ = f[0] * (MOTIF_MIRROR_NEXT / 0.5 if name == "mirror" and back == 1 else 1.0)
                fn = f[1]
                if first and not any(reachable(hh, fn(first[0]), t) for hh in ("R", "L")):
                    continue                             # the hands can't start this one here
                options.append((w_, name, fn))
            if not options:
                return
            name, fn = _pick(options, rng)
            swap = name in ("mirror",)                   # mirrored: the other hand plays it
            want = {pos: (tuple(sorted(fn(x) for x in bs)),
                          (other(hd) if swap else hd) if hd else None)
                    for pos, (bs, hd) in src.items() if bs}
            return

    def side_weight(b: int) -> float:
        """Lean towards the side that has had fewer notes lately."""
        if len(recent) < 6:
            return 1.0
        right = sum(1 for s in recent if s == "R") / len(recent) - 0.5
        return math.exp(-BALANCE_PULL * right if SIDE[b] == "R" else BALANCE_PULL * right)

    def slow_want(prev: int) -> int | None:
        p = slow_pat
        if p["name"] == "alternate":                     # across: 1 -> 8 -> 2 -> 7 ...
            return 9 - prev if rng.random() < 0.7 else (8 - prev) % 8 + 1
        if p["name"] == "walk":
            return (prev - 1 + p["dir"]) % 8 + 1
        if p["name"] == "repeat":
            return prev
        return None

    def choose(cands: list[tuple[str, int, float]]):
        """Weighted pick of (hand, button) from [(hand, button, weight)]."""
        total = sum(c[2] for c in cands)
        r = rng.random() * total
        for c in cands:
            r -= c[2]
            if r <= 0:
                return c[0], c[1]
        return cands[-1][0], cands[-1][1]

    def note_for(h: str, kind: str, k, t: float, button: int) -> Placed:
        at[h] = button
        hand_t[h] = t
        cont[h] = None                                   # only the next note is tied to its slide
        if kind == "slide":
            long = k.length >= 2 * (Fraction(1, 4) if 60.0 / beat < 120 else Fraction(1, 2))
            side = RIGHT if button <= 4 else LEFT
            opts = slide_options(button, side, k.restricted, long, allow_wifi)
            if not opts:
                return Placed("tap", button)
            shape, end = _pick(opts, rng)
            block(button, shape, end, k.clear_until, hand=h, ready=k.busy_until + next_sec)
            held[h] = button
            busy[h] = k.busy_until
            at[h] = end
            return Placed("slide", button, shape=shape, end=end)
        if kind == "hold":                               # its button gets room after release
            forbid.setdefault(button, []).append((k.busy_until + clear_sec, None))
            held[h] = button
            busy[h] = k.busy_until
            return Placed("hold", button)
        if kind == "touch":
            return Placed("touch", sensor=_touch_sensor(button, bool(strong and strong(t)), rng))
        return Placed("tap", button)

    hand_of_note = None

    def record(i: int, notes: list[Placed]):
        b_ = bar_of[i]
        btns = tuple(sorted(p.button for p in notes if p.button and not p.sub)) if notes else ()
        if beats is not None:
            played.setdefault(b_, {})[beats[i] - 4 * b_] = (btns, hand_of_note)
        for p in notes:
            if p.button:
                recent.append(SIDE[p.button])
        del recent[:-BALANCE_WINDOW]

    for i, (t, is_each) in enumerate(zip(times, each)):
        k = kinds[i] if kinds else None
        kind = k.kind if k else "tap"
        gap = t - last_t
        free = [hh for hh in ("R", "L") if not is_busy(hh, t)]
        if i == 0 or bar_of[i] != bar_of[i - 1]:
            start_bar(bar_of[i], t, beats[i] - 4 * bar_of[i] if beats is not None else 0)
        target, target_hand = want.get(beats[i] - 4 * bar_of[i], (None, None)) if want is not None else (None, None)
        hand_of_note = None

        # ---- burst: a very fast cluster, both hands
        if kind == "burst" and len(free) == 2:
            spacing = beat / k.burst_div
            notes = _burst(k.burst_n, at, rng, lambda b, j: ok(b, t + j * spacing),
                           allow_split=allow_wifi and 4 <= k.burst_n <= 5)
            if notes:
                end_t = t + (k.burst_n - 1) * spacing
                for p in notes:
                    hh = "R" if p.button <= 4 else "L"
                    at[hh], hand_t[hh] = p.button, t + p.sub * spacing
                out.append(notes)
                record(i, notes)
                last_hand, last_t, last_btn = ("R" if notes[-1].button <= 4 else "L"), end_t, notes[-1].button
                pattern["left"] = 0
                continue
            kind = "tap"                                 # no clean burst here: a plain note

        # ---- EACH: two buttons, one per hand
        if is_each and len(free) == 2:
            pairs = []
            for a in range(1, 9):
                for b in range(1, 9):
                    d = ring_distance(a, b)
                    if d == 0 or not (reachable("R", a, t) and reachable("L", b, t)):
                        continue
                    # the right hand takes the more clockwise-right button of the pair
                    if SIDE[a] == "L" and SIDE[b] == "R":
                        continue
                    near = 1.0 / (1 + ring_distance(at["R"], a) + ring_distance(at["L"], b))
                    wt = EACH_DIST[d] * near
                    if target is not None and set(target) == {a, b}:
                        wt *= MOTIF_BOOST                # the repeated bar's pair
                    elif target is not None and len(target) == 1 and target[0] in (a, b):
                        wt *= MOTIF_BOOST / 4            # a single note there before: keep it
                    pairs.append(("R", (a, b), wt))
            if pairs:
                _, (a, b) = choose(pairs)
                btn = {"R": a, "L": b}
                main = rng.choice("RL")
                o = other(main)
                notes = [note_for(main, kind, k, t, btn[main])]
                m = notes[0]
                double = (k and k.partner == kind and kind in ("hold", "slide") and m.kind == kind
                          and (kind != "slide" or ok(MIRROR[m.button], t)))
                if double and kind == "slide":           # mirrored double slide
                    notes.append(Placed("slide", MIRROR[m.button], MIRROR_SHAPE[m.shape], MIRROR[m.end]))
                    block(MIRROR[m.button], MIRROR_SHAPE[m.shape], MIRROR[m.end], k.clear_until,
                          hand=o, ready=k.busy_until + next_sec)
                    at[o], hand_t[o], held[o], busy[o] = MIRROR[m.end], t, MIRROR[m.button], k.busy_until
                elif double:
                    notes.append(note_for(o, "hold", k, t, btn[o]))
                else:
                    notes.append(note_for(o, "tap", None, t, btn[o]))
                out.append(notes)
                record(i, notes)
                last_hand, last_t, last_btn = o, t, btn[o]
                pattern["left"] = 0
                continue
            # no room for both: play it as a single note

        # ---- single note
        in_run = gap < FAST_SEC
        if in_run:
            if pattern["left"] <= 0:
                new_pattern()
            slow_pat["left"] = 0
        else:
            pattern["left"] = 0
            if slow_pat["left"] <= 0:
                name = _pick([(w_, n_) for w_, n_ in SLOW_PATTERNS], rng)[0]
                slow_pat.update(name=name, left=rng.randint(*PATTERN_LEN), dir=rng.choice((1, -1)))
        if target is not None and len(target) == 1:
            order_ = ([target_hand] if target_hand in free else []) + [hh for hh in free if hh != target_hand]
            pick = next((hh for hh in order_ if reachable(hh, target[0], t)
                         and not (in_run and len(free) == 2 and hh == last_hand
                                  and t - hand_t[hh] < ONE_HAND_SEC)), None)
            if pick:                                     # the repeated bar's note: take it
                hand_of_note = pick
                out.append([note_for(pick, kind, k, t, target[0])])
                record(i, out[-1])
                pattern["left"] = 0
                last_hand, last_t, last_btn = pick, t, target[0]
                continue
        w = weights(last_btn, gap, in_run)
        if not in_run:
            sw = slow_want(last_btn)
            if sw is not None:
                w[sw] *= SLOW_BOOST
        if target is not None:
            for tb in target:                           # (an EACH there before: either button)
                w[tb] *= MOTIF_BOOST / len(target)
        cands = []
        for hh in free:
            if in_run and len(free) == 2 and hh == last_hand and t - hand_t[hh] < ONE_HAND_SEC:
                continue                                 # fast: alternate hands
            for b in range(1, 9):
                if not reachable(hh, b, t):
                    continue
                wt = w[b] * side_weight(b)
                if not in_run and len(free) == 2 and SIDE[b] != hh:
                    wt *= 0.15                           # slow notes: the hand on that side
                if in_run and len(free) == 2 and hh == last_hand:
                    wt *= 0.2                            # (same hand again only if it has time)
                cands.append((hh, b, wt))
        if not cands:                                    # nowhere safe: leave this note out
            out.append([])
            record(i, [])
            continue
        hand, b = choose(cands)
        hand_of_note = hand
        out.append([note_for(hand, kind, k, t, b)])
        record(i, out[-1])
        if in_run:
            pattern["n"] += 1
            pattern["left"] -= 1
        else:
            slow_pat["left"] -= 1
        last_hand, last_t, last_btn = hand, t, b
    return out


def _burst(n: int, at: dict, rng: random.Random, free_at, allow_split: bool) -> list[Placed]:
    """Buttons for a burst of n slots: a swipe (neighbouring buttons in one
    direction, e.g. {24}1,2,3,4), a trill across the middle ({24}1,8,1,8) or a
    split (both hands sweeping apart, {32}8,1/7,2/6,3/5,4). [] if none fits."""
    kinds = [("swipe", 0.45), ("trill", 0.30)] + ([("split", 0.25)] if allow_split else [])
    rng.shuffle(kinds)
    kinds.sort(key=lambda kw: -kw[1] * rng.random())
    for name, _ in kinds:
        for _ in range(6):
            if name == "swipe":
                start = rng.choice((at["R"], at["L"]))
                d = rng.choice((1, -1))
                seq = [[(start - 1 + d * j) % 8 + 1] for j in range(n)]
            elif name == "trill":
                a = rng.choice(RIGHT)
                b = rng.choice((MIRROR[a], (a + 3) % 8 + 1))
                seq = [[a] if j % 2 == 0 else [b] for j in range(n)]
            else:                                         # split: L from 8 down, R from 1 down (or closing)
                m = n - 1
                left = [(8 - j - 1) % 8 + 1 for j in range(m)]      # 8, 7, 6, 5
                right = [j + 1 for j in range(m)]                   # 1, 2, 3, 4
                seq = [[left[0]]] + [[right[j], left[j + 1]] for j in range(m - 1)] + [[right[m - 1]]]
                if rng.random() < 0.5:                    # closing: from the bottom up
                    seq = [[5 - b if b <= 4 else 13 - b for b in s] for s in seq]   # top <-> bottom
            if all(free_at(b, j) for j, s in enumerate(seq) for b in s):
                return [Placed("tap", b, sub=j) for j, s in enumerate(seq) for b in s]
    return []