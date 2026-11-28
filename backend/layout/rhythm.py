"""When notes go (layout step 1): a note budget per bar, then the best slots.

  1. Rate every bar against the rest of the song (features.bar_matrix) and
     predict its density with the model learned from real charts
     (tools/fit_layout_tables.py): calm parts get fewer notes, like real
     charts, instead of a note on every peak.
  2. Turn that into a whole number of note moments per bar (rounding errors
     carried to the next bar, so the song's total is right).
  3. In each bar, take the best 16th slots: how likely real charts at this
     difficulty put a note there (place in the bar x hit strength x how busy
     the bar is), never closer than the difficulty allows.
  4. Decide which moments are EACH, holds or slides (shares from the tables;
     holds where the sound carries on, slides only where there is a long
     enough gap; nothing else is played during either, so both hands stay
     simple).

Times are 16th slots from beat 0 (the first downbeat): slot s = beat s / 4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from backend.checker.playability import ONE_HAND_SEC, slide_clear_beats
from backend.layout import features as F

SILENT_DB = -50.0                 # bars quieter than this (dB below the song's peak) are silence
# closest two moments may be, in 16th slots, by difficulty (and never under MIN_GAP_SEC)
MIN_GAP_SLOTS = {"basic": 4, "light": 2, "medium": 1, "dense": 1}
MIN_GAP_SEC = {"basic": 0.25, "light": 0.18, "medium": 0.12, "dense": 0.085}
EACH_ROOM_SEC = ONE_HAND_SEC      # an EACH needs this much room before and after
EACH_SHARE = 0.75                 # of the real charts' EACH share: a layout leaves some for the creator
HOLD_MIN_BEATS = Fraction(1, 2)
SUSTAIN_DB = 6.0                  # a hold where the sound stays within 6 dB over the gap
SLIDE_TRAVEL = (Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2))   # beats the star moves
LEVEL_OF = {"basic": "5", "light": "8", "medium": "11", "dense": "13"}


@dataclass
class Moment:
    slot: int                     # 16th slots from beat 0
    strength: float               # hit strength percentile there (0..1)
    each: bool = False
    hold: Fraction = Fraction(0)  # beats (0 = tap)
    slide: Fraction = Fraction(0) # star travel beats (0 = no slide)

    @property
    def beat(self) -> Fraction:
        return Fraction(self.slot, 4)


@dataclass
class Plan:
    moments: list[Moment]
    n_bars: int
    bar_target: np.ndarray        # wanted moments per bar
    bar_ratio: np.ndarray         # predicted density vs the song's average
    sound: np.ndarray             # bar has sound
    strength: np.ndarray          # per slot, percentile
    info: dict = field(default_factory=dict)


def group_for(density: str) -> str:
    if density not in LEVEL_OF:
        raise ValueError(f"density must be one of {list(LEVEL_OF)}")
    return density


def _round_carry(x: np.ndarray) -> np.ndarray:
    out, carry = np.zeros(len(x), np.int64), 0.0
    for i, v in enumerate(x):
        k = max(0, int(math.floor(v + carry + 0.5)))
        carry += v - k
        out[i] = k
    return out


def plan(mel_db: np.ndarray, bpm: float, first: float, duration: float, density: str, tables: dict,
         scale: float = 1.0) -> Plan:
    """Note moments for a single-BPM song. first: seconds of beat 0 (a
    downbeat); duration: song length in seconds."""
    g = group_for(density)
    beat_s = 60.0 / bpm
    n_bars = max(1, int((duration - first) / (4 * beat_s)))
    cv = F.curves(mel_db)
    t0 = first + np.arange(n_bars) * 4 * beat_s
    t1 = t0 + 4 * beat_s
    f0, f1 = (t0 * F.FPS).astype(np.int64), (t1 * F.FPS).astype(np.int64)
    loud_db = F._span_mean(cv["loud"], f0, f1)
    sound = loud_db > SILENT_DB
    idx = np.flatnonzero(sound)
    if len(idx) < 2:
        return Plan([], n_bars, np.zeros(n_bars), np.zeros(n_bars), sound, np.zeros(n_bars * 16))
    lo, hi = idx[0], idx[-1] + 1                              # charted stretch: first to last sounding bar

    slot_t = first + np.arange(n_bars * 16) * beat_s / 4
    raw = F.slot_strength(cv, slot_t)
    strength = np.zeros(n_bars * 16)
    strength[lo * 16:hi * 16] = F.percentile(raw[lo * 16:hi * 16])
    peaks = (strength >= 0.75).reshape(n_bars, 16).sum(1)

    X = F.bar_matrix(cv, t0[lo:hi], t1[lo:hi], peaks[lo:hi])
    ratio = np.zeros(n_bars)
    ratio[lo:hi] = F.predict(np.array(tables["model"][g]), X, tables.get("model_smooth", 1))
    ratio[~sound] = 0.0
    if ratio[lo:hi].mean() > 0:
        ratio[lo:hi] /= ratio[lo:hi].mean()
    k = int(np.searchsorted(tables["bpm_edges"], bpm, side="right"))
    per_bar = tables["density"][g]["per_bar_by_bpm"][k] or np.nanmean(
        [v for v in tables["density"][g]["per_bar_by_bpm"] if v])
    target = ratio * per_bar * scale
    counts = _round_carry(target)

    # ---- slots
    p = np.maximum(1e-4, np.array(tables["slots"][g]["p"]))     # [dens bin, metric, strength bin]
    dens_bin = np.searchsorted(tables["dens_edges"], ratio, side="right")
    metric = F.metric_class(np.arange(16))
    sbin = F.to_bin(strength)
    gap_slots = max(MIN_GAP_SLOTS[g], math.ceil(MIN_GAP_SEC[g] / (beat_s / 4) - 1e-9))
    taken = np.zeros(n_bars * 16, bool)
    picked: list[int] = []
    for b in range(lo, hi):
        if counts[b] <= 0:
            continue
        slots = np.arange(b * 16, b * 16 + 16)
        score = np.log(p[dens_bin[b], metric, sbin[slots]]) + 0.05 * strength[slots]
        n = 0
        for s in slots[np.argsort(-score, kind="stable")]:
            if n >= counts[b]:
                break
            a, z = max(0, s - gap_slots + 1), min(len(taken), s + gap_slots)
            if taken[a:z].any():
                continue
            taken[s] = True
            picked.append(int(s))
            n += 1
    picked.sort()
    moments = [Moment(s, float(strength[s])) for s in picked]
    _kinds(moments, ratio, tables, g, bpm, cv, first, beat_s)
    info = {"group": g, "bpm": bpm, "bars": n_bars, "charted_bars": int(hi - lo),
            "moments": len(moments), "per_bar_avg": per_bar * scale, "min_gap_slots": gap_slots}
    return Plan(moments, n_bars, target, ratio, sound, strength, info)


def _cell(tables, g, key, ratio_bar):
    """A density-table share for a bar, by how busy the bar is (averaged over
    the hit-strength bins of the loudness row its density matches best)."""
    tab = np.array(tables["density"][g][key])
    r = np.array(tables["density"][g]["ratio"])
    row = int(np.argmin(np.abs(r.mean(1) - ratio_bar)))
    return float(tab[row].mean())


def _kinds(moments: list[Moment], ratio, tables, g, bpm, cv, first, beat_s) -> None:
    if not moments:
        return
    clear = Fraction(slide_clear_beats(bpm)).limit_denominator(4)
    t = np.array([first + m.slot * beat_s / 4 for m in moments])
    gap_before = np.diff(t, prepend=-1e9)
    gap_after = np.append(np.diff(t), 1e9)
    nxt = [moments[i + 1].slot if i + 1 < len(moments) else None for i in range(len(moments))]

    by_bar: dict[int, list[int]] = {}
    for i, m in enumerate(moments):
        by_bar.setdefault(m.slot // 16, []).append(i)

    each_carry = hold_carry = slide_carry = 0.0
    for b, ids in sorted(by_bar.items()):
        r = float(ratio[b])
        # EACH: strongest moments on beats with room around them
        each_carry += EACH_SHARE * _cell(tables, g, "each", r) * len(ids)
        cand = sorted((i for i in ids if gap_before[i] >= EACH_ROOM_SEC and gap_after[i] >= EACH_ROOM_SEC
                       and moments[i].slot % 4 == 0), key=lambda i: -moments[i].strength)
        while each_carry >= 0.5 and cand:
            moments[cand.pop(0)].each = True
            each_carry -= 1.0
        # holds and slides only into an empty gap; nothing else plays meanwhile
        hold_carry += _cell(tables, g, "hold", r) * len(ids)
        slide_carry += _cell(tables, g, "slide", r) * len(ids)
        for i in sorted(ids, key=lambda i: -moments[i].strength):
            m = moments[i]
            if m.each or nxt[i] is None:
                continue
            gap = Fraction(nxt[i] - m.slot, 4)                     # beats to the next moment
            room = gap - clear
            if slide_carry >= 0.5 and room >= 1 + SLIDE_TRAVEL[0]:
                m.slide = max(x for x in SLIDE_TRAVEL if x <= room - 1)
                slide_carry -= 1.0
            elif hold_carry >= 0.5 and room >= HOLD_MIN_BEATS and _sustained(cv, t[i], t[i] + float(gap) * beat_s):
                m.hold = Fraction(math.floor(room * 2), 2)
                hold_carry -= 1.0
        each_carry = min(each_carry, 1.0); hold_carry = min(hold_carry, 1.0); slide_carry = min(slide_carry, 1.0)


def _sustained(cv, t0: float, t1: float) -> bool:
    f0 = int(t0 * F.FPS)
    f1 = max(f0 + 2, int(t1 * F.FPS))
    seg = cv["loud"][f0:f1]
    if len(seg) < 2:
        return False
    head = seg[: max(1, len(seg) // 5)].max()
    return float(np.median(seg)) >= head - SUSTAIN_DB
