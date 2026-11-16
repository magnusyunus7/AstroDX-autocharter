"""BPM + offset detection for constant-tempo songs.

Hard assumption (v1): the BPM is a whole number or ends in .5, .333 or .667.

Steps
  1. Onset envelopes (~2.9 ms resolution): percussive "attack" curve, a
     low-band (kick/bass) curve, and beat-level harmony change for downbeats.
  2. BPM: fold the attack curve over the whole song at every allowed BPM
     (60-420). The exact BPM keeps hits stacked in one place for the whole
     song; a BPM that's even 0.1 off smears them.
  3. Octave (157 vs 314): walk up the tempo family, accepting a faster tempo
     while its extra beats are clearly hit too. All family members are
     returned as candidates so the UI can offer them.
  4. Beat phase: the stacked peak of the fold, refined to sub-ms.
  5. Downbeat: which of the 4 beat positions gets the most kick/bass + the
     most harmonic change.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction

import numpy as np

VERSION = 5               # bump when results change, so cached detections are redone
HOP = 64                  # analysis hop (samples) -> 2.9 ms at 22050 Hz
COARSE = 4                # BPM search runs on a 4x coarser curve first
TOP_K = 12                # ...then re-scores this many best BPMs at full resolution
BPM_MIN, BPM_MAX = 50.0, 420.0
FRACTIONS = (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3))
# Onset curves peak slightly before/after the true attack. Calibrated on real
# charts (tools/eval_timing.py: three songs by different charters agreed on a
# +4.5 ms lag vs the -6 ms our synthetic drums suggested; real attacks are softer).
ONSET_BIAS_SEC = -0.0015
# Among octave candidates (x2, /2), prefer the fastest at or below this. On real
# charts, songs with gallops / 8th-note kicks fully hit the 2x grid too; every
# 2x mistake landed at 334-402 BPM while real fast charts were <= 314.
PREFERRED_MAX_BPM = 330.0


@dataclass
class Candidate:
    bpm: float
    score: float          # fold sharpness (higher = cleaner grid)
    relation: str         # "chosen", "x2", "/2", "x1.5", ...
    grid: dict = field(default_factory=dict)  # octave ratios per curve (debug/tuning)


@dataclass
class TimingResult:
    bpm: float
    first: float                      # time of a downbeat (earliest >= 0)
    beat_phase: float                 # time of the first beat (any beat) >= 0
    candidates: list[Candidate] = field(default_factory=list)
    confidence: dict = field(default_factory=dict)  # tempo / phase / downbeat in 0..1
    version: int = VERSION
    separate: bool = True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [asdict(c) for c in self.candidates]
        return d


# ------------------------------------------------------------------ helpers

def allowed_bpms(lo=BPM_MIN, hi=BPM_MAX) -> np.ndarray:
    out = [k + float(f) for k in range(int(lo), int(hi) + 1) for f in FRACTIONS]
    return np.array([b for b in out if lo <= b <= hi])


def snap_bpm(bpm: float) -> float | None:
    """Nearest allowed BPM if within 0.02, else None."""
    k = np.floor(bpm)
    for f in FRACTIONS:
        for base in (k, k + 1):
            c = base + float(f)
            if abs(c - bpm) < 0.02:
                return round(c, 4)
    return None


def _flux(S_db: np.ndarray) -> np.ndarray:
    d = np.diff(S_db, axis=1, prepend=S_db[:, :1])
    return np.maximum(d, 0).mean(axis=0)


def onset_curves(y: np.ndarray, sr: int, separate: bool = True):
    """Returns (attack, body, lowband, chroma, fps). Frame i is centered at i*HOP/sr.

    attack: all percussive attacks, used for exact BPM and phase.
    body:   percussive attacks below ~2.5 kHz (kick, snare body, bass); ignores
            hi-hats, so offbeat hats don't make the tempo look twice as fast.

    separate: split drums from melody (HPSS) first. ~3-4x slower, but on real
    charts it fixed a wrong BPM family (fast mode: 8/9 right, separate: 9/9).
    """
    import librosa
    y_p = librosa.effects.hpss(y, margin=2.0)[1] if separate else y
    mel = librosa.feature.melspectrogram(y=y_p, sr=sr, n_fft=512, hop_length=HOP,
                                         n_mels=64, fmax=sr / 2, power=2.0)
    attack = _flux(librosa.power_to_db(mel, ref=np.max, top_db=80))
    bmel = librosa.feature.melspectrogram(y=y_p, sr=sr, n_fft=1024, hop_length=HOP,
                                          n_mels=40, fmax=2500, power=2.0)
    body = _flux(librosa.power_to_db(bmel, ref=np.max, top_db=80))

    lowmel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=HOP,
                                            n_mels=128, fmax=sr / 2, power=2.0)
    low = _flux(librosa.power_to_db(lowmel[:12], ref=np.max, top_db=80))  # lowest ~12 mel bands

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096, hop_length=HOP * 8)
    win = int(1.0 * sr / HOP)
    attack, body, low = (np.maximum(x - _moving_avg(x, win), 0) for x in (attack, body, low))
    return attack, body, low, chroma, sr / HOP


def _moving_avg(x, n):
    n = max(1, n)
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def fold(env: np.ndarray, period_frames: float, nbins: int) -> np.ndarray:
    """Average of env at each phase bin of a period (in frames)."""
    ph = (np.arange(len(env)) / period_frames) % 1.0
    idx = np.minimum((ph * nbins).astype(np.int64), nbins - 1)
    s = np.bincount(idx, weights=env, minlength=nbins)
    c = np.bincount(idx, minlength=nbins)
    return s / np.maximum(c, 1)


def _sharpness(h: np.ndarray) -> float:
    hs = (h + np.roll(h, 1) + np.roll(h, -1)) / 3
    sd = h.std() or 1e-9
    return float((hs.max() - h.mean()) / sd)


# ------------------------------------------------------------------ detection

def detect(y: np.ndarray, sr: int, start_sec: float = 0.0, separate: bool = True) -> TimingResult:
    attack, body, low, chroma, fps = onset_curves(y, sr, separate)
    s0 = int(max(0.0, start_sec) * fps)
    env = attack.copy()
    env[:s0] = 0
    body = body.copy()
    body[:s0] = 0

    # --- 1. exact BPM: fold sharpness over all allowed BPMs on a coarse curve,
    #        then re-score the best few at full resolution
    bpms = allowed_bpms()
    n = len(env) // COARSE * COARSE
    coarse = env[:n].reshape(-1, COARSE).max(axis=1)
    scores = np.array([_sharpness(fold(coarse, 60 / b * fps / COARSE, 48)) for b in bpms])
    top = bpms[np.argsort(scores)[::-1][:TOP_K]]
    fine = {float(b): _sharpness(fold(env, 60 / b * fps, 48)) for b in top}
    best = max(fine, key=fine.get)

    # --- 2. octave family of the best BPM
    def score_of(b):
        return _sharpness(fold(env, 60 / b * fps, 48))

    family = {}
    for num, den, rel in [(1, 4, "/4"), (1, 3, "/3"), (1, 2, "/2"), (2, 3, "x2/3"), (1, 1, "x1"),
                          (3, 2, "x1.5"), (2, 1, "x2"), (3, 1, "x3"), (4, 1, "x4")]:
        b = snap_bpm(best * num / den)
        if b is not None and BPM_MIN <= b <= BPM_MAX:
            family[b] = rel

    low_c = low.copy()
    low_c[:s0] = 0
    ratios = octave_ratios({"body": body, "low": low_c}, fps, family)
    chosen = _choose_octave(ratios)

    # always offer exactly half and double of the chosen BPM (e.g. a chart
    # written at 57.333 for a song detected at 114.667)
    for b in (snap_bpm(chosen / 2), snap_bpm(chosen * 2)):
        if b is not None and BPM_MIN <= b <= BPM_MAX and b not in family:
            family[b] = ""
            ratios[b] = {}
    candidates = sorted(
        (Candidate(b, round(score_of(b), 3), "chosen" if b == chosen else _rel(b, chosen),
                   ratios[b]) for b in family),
        key=lambda c: (c.relation != "chosen", -c.score))

    # --- 3. beat phase (sub-ms): fold at the chosen period with fine bins
    period = 60 / chosen
    phase = _beat_phase(env, fps, period) - ONSET_BIAS_SEC
    phase %= period

    # --- 4. downbeat: which beat of 4
    db_off, db_conf = _downbeat(low, chroma, fps, period, phase, start_sec)
    first = (phase + db_off * period) % (4 * period)

    # confidences
    srt = np.sort(scores)[::-1]
    tempo_conf = float(np.clip((srt[0] - np.median(scores)) / (srt[0] + 1e-9), 0, 1))
    return TimingResult(
        bpm=chosen, first=round(first, 4), beat_phase=round(phase, 4), candidates=candidates,
        confidence={"tempo": round(tempo_conf, 3), "downbeat": round(db_conf, 3)})


def _rel(b, chosen) -> str:
    r = b / chosen
    for v, name in [(0.25, "/4"), (1 / 3, "/3"), (0.5, "/2"), (2 / 3, "x2/3"), (1.5, "x1.5"),
                    (2, "x2"), (3, "x3"), (4, "x4")]:
        if abs(r - v) < 1e-3:
            return name
    return f"x{r:.2f}"


def _grid_hit_strength(env, fps, bpm) -> float:
    """How strongly *every* beat of this tempo's grid is hit: the height of the
    folded peak above the median. A tempo whose grid includes unhit positions
    (e.g. 2x the true tempo, adding offbeats) averages them in and scores lower."""
    h = fold(env, 60 / bpm * fps, 96)
    h = (h + np.roll(h, 1) + np.roll(h, -1)) / 3
    return float(h.max() - np.median(h))


# A tempo is "fully hit" when it keeps at least this share of the best grid
# strength in each curve. Tuned on synthetic songs (right tempo: low >= 0.73,
# body >= 0.65; 2x too fast: low <= 0.64). Retune with tools/eval_timing.py --debug.
OCTAVE_KEEP = {"body": 0.60, "low": 0.68}


def octave_ratios(envs: dict, fps, family) -> dict:
    """{bpm: {"body": r, "low": r}}: each member's grid strength relative to
    the strongest member, per curve."""
    out = {b: {} for b in family}
    for name, env in envs.items():
        strength = {b: _grid_hit_strength(env, fps, b) for b in family}
        top = max(strength.values()) or 1e-9
        for b in family:
            out[b][name] = round(strength[b] / top, 3)
    return out


def _is_octave_of(b: float, ref: float) -> bool:
    k = np.log2(b / ref)
    return abs(k - round(k)) < 1e-3


def _choose_octave(ratios: dict) -> float:
    """Anchor on the candidate whose grid is strongest in *both* curves, then
    among its true octaves (x2, /2, x4...) pick the fastest whose beats are
    all clearly hit, staying at or below PREFERRED_MAX_BPM when possible.
    Cross-rhythms (x2/3, x1.5) are never chosen automatically; they're
    offered as candidates."""
    anchor = max(ratios, key=lambda b: min(ratios[b].values()))
    chain = sorted(b for b in ratios if _is_octave_of(b, anchor))
    ok = [b for b in chain if all(ratios[b][k] >= OCTAVE_KEEP[k] for k in OCTAVE_KEEP)]
    within = [b for b in ok if b <= PREFERRED_MAX_BPM]
    if within:
        return max(within)
    if ok:
        return min(ok)
    under = [b for b in chain if b <= PREFERRED_MAX_BPM] or chain
    return max(under, key=lambda b: min(ratios[b][k] / OCTAVE_KEEP[k] for k in OCTAVE_KEEP))


def _peak_near(h, idx, width=3):
    n = len(h)
    return max(h[int(round(idx + d)) % n] for d in range(-width, width + 1))


def _beat_phase(env, fps, period) -> float:
    nb = 400
    h = fold(env, period * fps, nb)
    h = h - np.median(h)
    i = int(np.argmax(h))
    # circular weighted centroid of the peak (+-4% of the beat)
    w = max(2, nb // 25)
    idx = np.arange(i - w, i + w + 1)
    vals = np.maximum(h[idx % nb], 0)
    c = (idx * vals).sum() / (vals.sum() or 1)
    return (c + 0.5) / nb * period


def _downbeat(low, chroma, fps, period, phase, start_sec):
    """Score each of the 4 beat positions; return (offset in beats 0..3, confidence)."""
    n_beats = int((len(low) / fps - phase) / period)
    beats = phase + np.arange(n_beats) * period
    beats = beats[beats >= start_sec]
    if len(beats) < 16:
        return 0, 0.0
    tol = max(1, int(0.03 * fps))

    def at(env, t):
        i = int(round(t * fps))
        return env[max(0, i - tol): i + tol + 1].max(initial=0)

    lowv = np.array([at(low, t) for t in beats])
    # harmony change: cosine distance of chroma just before vs just after each beat
    cfps = fps / 8
    ch = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    span = max(1, int(period * cfps * 0.9))

    def hchange(t):
        i = int(round(t * cfps))
        if i - span < 0 or i + span > ch.shape[1]:
            return 0.0
        a = ch[:, i - span: i].mean(axis=1)
        b = ch[:, i: i + span].mean(axis=1)
        return 1 - float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))

    hv = np.array([hchange(t) for t in beats])
    first_idx = int(round((beats[0] - phase) / period))

    def norm(v):
        return (v - v.mean()) / (v.std() + 1e-9)

    combined = norm(lowv) + norm(hv)
    sc = np.zeros(4)
    for j in range(4):
        sc[j] = combined[(np.arange(len(beats)) + first_idx) % 4 == j].mean()
    order = np.argsort(sc)[::-1]
    conf = float(np.clip((sc[order[0]] - sc[order[1]]) / (abs(sc[order[0]]) + 1e-9), 0, 1))
    return int(order[0]), conf