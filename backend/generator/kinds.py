"""What kind of note each moment becomes: tap, hold, slide or touch, plus EACH
and BREAK.

Placement decided *when* things are hit. Here some of those moments turn into
notes that keep a hand busy (holds, slides) or move to the touch screen:

  * slide: star tap, the star waits 1 beat (simai default), then travels for
    `length` (mostly [8:1]); the hand is busy from the tap to the end of the
    travel. Then finish room (at least a {4}, more at fast BPM): no EACH and
    no note on the slide's path, or the finish "splashes" into it
  * hold: the hand stays down for `length`; a long gap after the note (the
    sound sustains) makes a natural hold
  * touch: a single note moved to the touch screen (sensor chosen with the
    buttons, in hands.py)

While one hand is busy the other hand plays every note in that window alone.
A hold/slide may drop a few weak notes so that's one-hand speed (charters thin
out around slides too); the pipeline plans a little extra density to make up.
Busy windows never contain an EACH: the strongest EACH are placed first, the
rest after the holds/slides, around them.
Counts follow the level targets (slides per minute, hold %, touch %, BREAK %).

Order: half the EACH -> slides and holds (taking turns) -> drop -> rest of EACH
-> touch -> BREAK.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from backend.checker.playability import slide_clear_beats
from backend.generator.difficulty import Targets, level_value
from backend.generator.placement import SUB, Moment, mark_each_moments

ONE_HAND_SEC = 0.15          # one hand alone: notes at least this far apart
RELEASE_SEC = 0.06           # a busy hand is free again this long after its note ends
FAST_SEC = 0.20              # same as hands.FAST_SEC: closer notes alternate hands
HOLD_BEATS = (Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4))
HOLD_SEC = (0.20, 1.6)       # hold length range (seconds)
NATURAL_GAP_BEATS = 1        # a gap this long after a note = sustained sound -> hold
# slide travel, in units of [8:1] (half a beat; [16:1] on slow songs that are
# charted in double time). Real charts: ~75% [8:1], never faster.
SLIDE_UNITS = ((1, 0.72), (2, 0.14), (3, 0.06), (4, 0.08))     # (units, share)
DOUBLE_TIME_BPM = 120        # below: one unit = [16:1]
SLIDE_WAIT = Fraction(1)     # star waits 1 beat before moving (simai default)
EACH_FIRST_SHARE = 0.5       # EACH placed before holds/slides (the rest after, around them)
EACH_ROOM_SEC = 0.09         # an EACH needs at least this much room on both sides
TOUCH_GAP_SEC = 0.08         # touch notes need this much room on both sides
BREAK_ROOM_SEC = 0.20        # BREAK accents: this much room on both sides
BREAK_SPACING_SEC = 1.0      # ...and at least this far from other BREAKs
MAX_DROP = {"hold": 2, "slide": 3}
HOLD_KEEP_GUESS = 0.75       # share of notes left after the drops (hold % is counted on those)   # notes a hold/slide may remove to free the other hand


@dataclass
class Kind:
    kind: str = "tap"                 # tap / hold / slide / touch
    length: Fraction = Fraction(0)    # hold length or slide travel, in beats
    partner: str = "tap"              # EACH partner: tap, or the same kind mirrored
    restricted: bool = False          # slide: other hand plays during it -> stay in own half
    brk: bool = False
    busy_until: float = 0.0           # time the hand is free again
    first_other: float = 1e9          # time of the first note the other hand plays meanwhile
    clear_until: float = 0.0          # slide: until then, no EACH and nothing on its path


def decide(moments: list[Moment], times: list[float], analysis: dict, targets: Targets,
           gap_steps: int, seed: int = 0, break_grids: set[int] | None = None,
           end_limit: float = float("inf")) -> tuple[list[Moment], list[float], list[Kind]]:
    """Moments without EACH in; kept moments (EACH marked), their times and kinds out."""
    rng = random.Random(seed * 7919 + 17)
    n = len(moments)
    if n == 0:
        return [], [], []
    out = [Kind() for _ in range(n)]
    beat = 60.0 / analysis["bpm"]
    t = np.asarray(times, float)
    each_frac = targets.each_pct / 100.0
    span_min = max(1e-6, (t[-1] - t[0]) / 60.0)
    lv = level_value(targets.level)
    pos = np.array([m.grid % SUB for m in moments])
    strength = np.array([m.strength for m in moments])
    gap_next = np.append(np.diff(t), 1e9)
    owner = np.full(n, -1)            # moment -> the hold/slide whose window it's in
    removed = np.zeros(n, bool)
    noeach = np.zeros(n, bool)       # inside a slide's finish room: no EACH here

    # ---- EACH, part 1: the strongest share first; holds/slides work around them
    step = beat / SUB
    each_room = max(gap_steps, int(np.ceil(EACH_ROOM_SEC / step - 1e-9)))
    mark_each_moments(moments, int(round(each_frac * n * EACH_FIRST_SHARE)), gap_steps,
                      min_room=each_room)
    each = np.array([m.each for m in moments])

    def fits(i: int, end_t: float, max_drop: int):
        """(notes the other hand plays alone, notes to drop) while i's hand is
        busy until end_t; None if that needs more than max_drop drops."""
        if owner[i] >= 0 or removed[i]:
            return None
        w = []
        j = i + 1
        while j < n and t[j] < end_t + RELEASE_SEC:
            if not removed[j]:
                if owner[j] >= 0 or each[j]:               # an EACH needs both hands
                    return None
                w.append(j)
            j += 1
        drop = []
        if each[i]:                                     # partner taps at t[i] with the other hand
            lead = [t[i]]
        elif i > 0 and not removed[i - 1] and t[i] - t[i - 1] < FAST_SEC:
            lead = [t[i - 1]]                           # the other hand just played i-1
        else:
            lead = []
        while True:
            seq = lead + [t[x] for x in w]
            bad = [k for k, (a, b) in enumerate(zip(seq, seq[1:])) if b - a < ONE_HAND_SEC]
            if not bad:
                break
            k = bad[0] - len(lead)                      # pair (k, k+1) in w; k = -1: the lead note
            cand = [x for x in (k, k + 1) if x >= 0]
            x = min(cand, key=lambda x: strength[w[x]])
            drop.append(w.pop(x))
            if len(drop) > max_drop:
                return None
        return w, drop

    def claim(i: int, kind: str, length: Fraction, fit, end_t: float):
        w, drop = fit
        removed[drop] = True
        k = out[i]
        k.kind, k.length, k.busy_until = kind, length, end_t
        k.restricted = bool(w)
        k.first_other = t[w[0]] if w else 1e9
        owner[i] = i
        owner[w] = i

    def next_time(i: int, cond) -> float:
        j = i + 1
        while j < n and (removed[j] or not cond(j)):
            j += 1
        return t[j] if j < n else float("inf")

    def try_lengths(i, kind, lengths, extra_beats=Fraction(0), tail=Fraction(0)):
        """First length that fits: the hand is free before the next busy window,
        `tail` beats of room are left before the next EACH (slide finish), and
        it all ends before the audio does."""
        window_at = next_time(i, lambda j: owner[j] >= 0)
        each_at = next_time(i, lambda j: each[j])
        for L in lengths:
            end_t = t[i] + float(extra_beats + L) * beat
            clear_t = end_t + float(tail) * beat
            if end_t + RELEASE_SEC > window_at or clear_t > each_at + 1e-6 or clear_t > end_limit:
                continue
            fit = fits(i, end_t, MAX_DROP[kind])
            if fit is not None:
                claim(i, kind, L, fit, end_t)
                out[i].clear_until = clear_t
                # notes while it finishes come from the other side: stay in own half
                out[i].restricted = out[i].restricted or next_time(i, lambda j: True) < clear_t
                if tail:                                # no EACH while the slide finishes
                    noeach[(t > t[i]) & (t < clear_t)] = True
                return True
        return False

    # ---- slides and holds, taking turns (whichever is further behind its
    # target goes next) so neither uses up all the room for the other
    n_slide = int(round(targets.slides_per_min * span_min))
    n_hold = int(round(targets.hold_pct / 100.0 * n * (1 + each_frac) * HOLD_KEEP_GUESS))
    unit = Fraction(1, 4) if analysis["bpm"] < DOUBLE_TIME_BPM else Fraction(1, 2)
    clear = Fraction(slide_clear_beats(analysis["bpm"])).limit_denominator(4)
    hold_lengths = [L for L in HOLD_BEATS if HOLD_SEC[0] <= float(L) * beat <= HOLD_SEC[1]] or [HOLD_BEATS[0]]
    natural = gap_next >= NATURAL_GAP_BEATS * beat
    # stars on strong beats; holds where the sound sustains, else on strong beats
    slide_order = sorted(range(n), key=lambda i: -(
        strength[i] + (0.3 if pos[i] == 0 else 0.1 if pos[i] == 6 else 0.0) + rng.uniform(0, 0.5)))
    hold_order = sorted(range(n), key=lambda i: -(
        strength[i] + (0.4 if natural[i] else 0.0)
        + (0.15 if pos[i] == 0 else 0.05 if pos[i] == 6 else 0.0) + rng.uniform(0, 0.35)))

    def slide_at(i):
        # a weighted pick, then the plain [8:1] if that doesn't fit; the hand
        # stays busy until `clear` after the slide ends
        r, units = rng.random(), 1
        for u, share in SLIDE_UNITS:
            r -= share
            if r <= 0:
                units = u
                break
        ls = [unit * units] + ([unit] if units > 1 else [])
        return try_lengths(i, "slide", ls, SLIDE_WAIT, tail=clear)

    def hold_at(i):
        if natural[i]:                                   # longest that releases before the next note
            ls = sorted((L for L in hold_lengths if float(L) * beat <= gap_next[i] - 0.5 * beat),
                        key=float, reverse=True)[:2] or hold_lengths[:1]
        else:                                            # random length that fits, else shorter
            ls = hold_lengths[:]
            rng.shuffle(ls)
            ls = ls[:2] + sorted(ls[2:], key=float)
        return try_lengths(i, "hold", ls)

    todo = {"slide": [slide_order, 0, 0, max(1, n_slide), slide_at],    # order, next, made, want, fn
            "hold": [hold_order, 0, 0, max(1, n_hold), hold_at]}
    if n_slide == 0:
        del todo["slide"]
    if n_hold == 0:
        del todo["hold"]
    while todo:
        name = min(todo, key=lambda k: todo[k][2] / todo[k][3])
        st = todo[name]
        if st[2] >= st[3] or st[1] >= n:
            del todo[name]
            continue
        st[2] += st[4](st[0][st[1]])
        st[1] += 1

    # ---- drop the notes that made room
    keep = np.nonzero(~removed)[0]
    moments = [moments[i] for i in keep]
    out = [out[i] for i in keep]
    t, pos, strength = t[keep], pos[keep], strength[keep]
    is_head = np.array([k.kind in ("hold", "slide") for k in out])
    n = len(moments)

    # ---- EACH, part 2: the rest, outside busy windows (a hold/slide head may
    # be one if the other hand has time before its next note)
    owner_kept, noeach_kept = owner[keep], noeach[keep]
    want = int(round(each_frac * n))
    have = sum(m.each for m in moments)
    if have < want:
        allowed = [not moments[i].each and not noeach_kept[i] and (owner_kept[i] < 0 or (
            is_head[i] and out[i].first_other - t[i] >= ONE_HAND_SEC)) for i in range(n)]
        mark_each_moments(moments, want - have, gap_steps, allowed, min_room=each_room)
    elif have > want:                                    # the drops were single notes
        plain = [i for i in range(n) if moments[i].each and not is_head[i]]
        for i in sorted(plain, key=lambda i: strength[i])[:have - want]:
            moments[i].each = False
    for i in range(n):
        quiet_until = max(out[i].busy_until + RELEASE_SEC, out[i].clear_until)
        nothing_after = i + 1 >= n or t[i + 1] >= quiet_until   # both hands busy: nothing may come
        if moments[i].each and is_head[i] and nothing_after \
                and rng.random() < (0.35 if lv >= 13 else 0.15):
            out[i].partner = out[i].kind                 # double hold / mirrored double slide
    hits = sum(2 if m.each else 1 for m in moments)

    # ---- touch: single taps with a little room
    gap_prev = np.insert(np.diff(t), 0, 1e9)
    gap_next = np.append(np.diff(t), 1e9)
    n_touch = int(round(targets.touch_pct / 100.0 * hits))
    in_slide = np.zeros(n, bool)                         # touch sensors sit on slide paths
    for i in range(n):
        if out[i].kind == "slide":
            in_slide |= (t > t[i]) & (t < out[i].clear_until)
    cand = [i for i in range(n) if out[i].kind == "tap" and not moments[i].each and not in_slide[i]
            and gap_prev[i] >= TOUCH_GAP_SEC and gap_next[i] >= TOUCH_GAP_SEC]
    rng.shuffle(cand)
    for i in cand[:n_touch]:
        out[i].kind = "touch"

    # ---- BREAK: section starts, then the strongest isolated accents (not touch)
    n_break = int(round(targets.break_pct / 100.0 * hits))
    got = 0
    for i, m in enumerate(moments):
        if break_grids and m.grid in break_grids and out[i].kind != "touch":
            out[i].brk = True
            got += 2 if m.each else 1
    room = np.minimum(gap_prev, gap_next)
    # roomy accents first, then anything (a BREAK inside a run is fine, just rarer)
    order = sorted((i for i in range(n) if out[i].kind != "touch" and not out[i].brk),
                   key=lambda i: (room[i] < BREAK_ROOM_SEC,
                                  -(strength[i] + (0.3 if pos[i] == 0 else 0.0) + rng.uniform(0, 0.2))))
    taken = sorted(t[i] for i in range(n) if out[i].brk)
    for i in order:
        if got >= n_break:
            break
        k = np.searchsorted(taken, t[i])
        if (k > 0 and t[i] - taken[k - 1] < BREAK_SPACING_SEC) or \
                (k < len(taken) and taken[k] - t[i] < BREAK_SPACING_SEC):
            continue                                     # spread them out
        out[i].brk = True
        taken.insert(k, t[i])
        got += 2 if moments[i].each else 1
    return moments, [float(x) for x in t], out