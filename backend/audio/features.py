"""Phase 2 analysis on the confirmed timing grid.

grid onsets: for every grid position (1/12 of a beat, which covers 1/4, 1/8,
             1/16 and triplets), how strongly something is hit there, per stem:
               drums  - percussive attacks (kick, snare, hats)
               bass   - low-frequency attacks (kick body, bass notes)
               melody - new pitches (chroma change: leads, vocals, chords).
                        Repeating the same note doesn't register (v1 limit).
bar fit:     per bar, how far the hits sit from the grid. A run of bars that
             are off means the tempo changes there (or the BPM is slightly off).
"""
from __future__ import annotations

import math

import numpy as np

VERSION = 7
HOP = 64                  # ~2.9 ms at 22050 Hz
SUBDIV = 12               # grid points per beat (lcm of 4 and 3)
STEMS = ("drums", "bass", "melody")
TOL_SEC = 0.025           # a hit counts for a grid point within +-25 ms
# Each onset curve peaks slightly before the true attack (window lengths
# differ). Median peak - truth, measured on synthetic songs with known event
# times (tools/make_test_song.py events); curves are shifted to compensate.
BIAS_SEC = {"drums": -0.0068, "bass": -0.0171, "melody": -0.0265}

DRIFT_MS = 22.0           # a bar whose hits sit further than this from the grid is "off"
LOW_CONTRAST = 0.45       # ...or whose hits are this much less grid-locked than usual
CLEAR_CONTRAST = 0.8      # offsets are only trusted in bars at least this grid-locked
                          # (in weaker bars the peak, hence the offset, is mostly noise)
QUIET_ENERGY = 0.5        # bars quieter than this x typical are not judged (breakdowns)
DRIFT_RUN = 3             # this many consecutive off bars -> warning


def _flux(S_db: np.ndarray) -> np.ndarray:
    d = np.diff(S_db, axis=1, prepend=S_db[:, :1])
    return np.maximum(d, 0).mean(axis=0)


def _detrend(x: np.ndarray, n: int) -> np.ndarray:
    k = np.ones(max(1, n)) / max(1, n)
    return np.maximum(x - np.convolve(x, k, mode="same"), 0)


def _shift(x: np.ndarray, frames: int) -> np.ndarray:
    """out[i] = x[i + frames] (zero-filled), i.e. move peaks `frames` later when frames < 0."""
    if frames == 0:
        return x
    out = np.zeros_like(x)
    if frames > 0:
        out[:-frames] = x[frames:]
    else:
        out[-frames:] = x[:frames]
    return out


def stem_curves(y: np.ndarray, sr: int) -> tuple[dict[str, np.ndarray], float]:
    """Onset strength curves per stem, bias-corrected so peaks sit on the
    true attack. Frame i is at time i*HOP/sr."""
    import librosa
    _, y_p = librosa.effects.hpss(y, margin=2.0)

    def logmel(sig, n_fft, **kw):
        m = librosa.feature.melspectrogram(y=sig, sr=sr, n_fft=n_fft, hop_length=HOP, power=2.0, **kw)
        return librosa.power_to_db(m, ref=np.max, top_db=80)

    fps = sr / HOP
    n = 1 + len(y) // HOP
    # melody: chroma flux at a 4x coarser hop (memory), interpolated back
    ch_hop = HOP * 4
    ch = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=ch_hop)
    chf = np.maximum(np.diff(ch, axis=1, prepend=ch[:, :1]), 0).sum(axis=0)
    melody = np.interp(np.arange(n) * HOP, np.arange(len(chf)) * ch_hop, chf)

    raw = {
        "drums": _flux(logmel(y_p, 512, n_mels=64)),
        "bass": _flux(logmel(y, 1024, n_mels=8, fmin=30, fmax=250)),  # ~10 FFT bins here: 8 bands
        "melody": melody,
    }
    win = int(1.0 * fps)
    return {k: _shift(_detrend(v[:n], win), int(round(BIAS_SEC[k] * fps)))
            for k, v in raw.items()}, fps


def _windowed_max(x: np.ndarray, half: int) -> np.ndarray:
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(x, size=2 * half + 1, mode="constant")


def grid_layout(bpm: float, first: float, duration: float) -> tuple[int, int]:
    """(k_start, n_beats): grid starts at the bar line at or before 0 s and
    covers the whole song. Beat index k is relative to `first` (k=0 is beat 1)."""
    beat = 60.0 / bpm
    k0 = math.ceil((0.0 - first) / beat - 1e-9)
    k_start = math.floor(k0 / 4) * 4
    k_end = math.ceil((duration - first) / beat)
    n_beats = int(math.ceil((k_end - k_start) / 4) * 4)
    return k_start, n_beats


def grid_onsets(curves, fps, bpm, first, duration) -> dict:
    beat = 60.0 / bpm
    k_start, n_beats = grid_layout(bpm, first, duration)
    times = first + (k_start + np.arange(n_beats * SUBDIV) / SUBDIV) * beat
    step = beat / SUBDIV
    half = max(1, int(round(min(TOL_SEC, step * 0.45) * fps)))
    out = {}
    for name in STEMS:
        env = curves[name]
        wm = _windowed_max(env, half)
        idx = np.round(times * fps).astype(int)
        valid = (idx >= 0) & (idx < len(env)) & (times >= 0)
        v = np.zeros(len(times))
        v[valid] = wm[idx[valid]]
        # scale: 1.0 = a typical strong hit (95th percentile of the 16th positions)
        sixteenths = v.reshape(-1, SUBDIV)[:, ::SUBDIV // 4].ravel()
        ref = np.percentile(sixteenths[sixteenths > 0], 95) if (sixteenths > 0).any() else 1.0
        out[name] = np.clip(v / (ref or 1.0), 0, 2.0)
    return {"k_start": k_start, "n_beats": n_beats, "subdiv": SUBDIV, "stems": out}


GRID_POINTS = (0.0, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0)   # 16ths and triplets within a beat
OTHER_TEMPO = 1.6         # a nearby BPM must fit this much better to call a bar "off",
                          # and give a clear grid itself (>= CLEAR_CONTRAST x typical):
                          # trying ~60 tempos on loose rhythm always finds a slightly better one


def _fold_beat(env, frame_t, i0, i1, first, period, nb=96):
    ph = ((frame_t[i0:i1] - first) / period) % 1.0
    h = np.bincount((ph * nb).astype(int) % nb, weights=env[i0:i1], minlength=nb)
    return (h + np.roll(h, 1) + np.roll(h, -1)) / 3


def _contrast(h):
    return float((h.max() - np.median(h)) / (h.mean() + 1e-9))


def bar_fit(curves, fps, bpm, first, k_start, n_beats) -> list[dict]:
    """Per bar (with half a bar of context either side):
      offset_ms  how far the strongest hits sit from the nearest 16th or
                 triplet position (+ = late)
      contrast   how grid-locked the hits are
      status     ok / off (hits don't fit this tempo/offset here) /
                 irregular (rhythm too loose to judge: rolls, glitches,
                 free-time intros) / quiet (breakdown, silence)"""
    beat = 60.0 / bpm
    env = curves["drums"] + curves["bass"]
    n_bars = n_beats // 4
    frame_t = np.arange(len(env)) / fps
    rows, bar_energy = [], []
    for b in range(n_bars):
        t0 = first + (k_start + 4 * b - 2) * beat
        i0, i1 = max(0, int(t0 * fps)), min(len(env), int((t0 + 8 * beat) * fps))
        if i1 - i0 < 8:
            rows.append(None); bar_energy.append(0.0); continue
        h = _fold_beat(env, frame_t, i0, i1, first, beat)
        frac = (int(np.argmax(h)) + 0.5) / len(h)
        nearest = min(GRID_POINTS, key=lambda g: abs(frac - g))
        off = (frac - nearest) * beat
        c0, c1 = max(0, int((t0 + 2 * beat) * fps)), min(len(env), int((t0 + 6 * beat) * fps))
        e_bar = float(env[c0:c1].mean()) if c1 > c0 else 0.0
        rows.append({"offset_ms": round(off * 1000, 1), "contrast": round(_contrast(h), 2),
                     "energy": round(e_bar, 4), "i0": i0, "i1": i1})
        bar_energy.append(e_bar)

    e = np.array(bar_energy)
    loud = np.median(e[e > 0]) if (e > 0).any() else 1.0
    judged = [r["contrast"] for b, r in enumerate(rows) if r and e[b] >= QUIET_ENERGY * loud]
    typical = float(np.median(judged)) if judged else 1.0

    def other_tempo_fits(r) -> bool:
        """Does a nearby different BPM (+-15%) fit this stretch much better?
        That's a tempo change; if nothing fits, the rhythm is just irregular."""
        # widen to 4 bars for tempo resolution
        mid = (r["i0"] + r["i1"]) // 2
        half = int(8 * beat * fps)
        i0, i1 = max(0, mid - half), min(len(env), mid + half)
        here = _contrast(_fold_beat(env, frame_t, i0, i1, first, beat))
        best = 0.0
        for f in np.arange(0.85, 1.151, 0.005):
            if abs(f - 1) < 0.004:
                continue
            p = beat / f
            # phase-free: fold anchored at the window start
            best = max(best, _contrast(_fold_beat(env, frame_t, i0, i1, frame_t[i0], p)))
        return best > OTHER_TEMPO * here and best >= CLEAR_CONTRAST * typical

    bars = []
    for b, r in enumerate(rows):
        if r is None or e[b] < QUIET_ENERGY * loud:
            status = "quiet"
        elif r["contrast"] < LOW_CONTRAST * typical:
            # hits aren't grid-locked: a different tempo here, or just loose rhythm?
            status = "off" if other_tempo_fits(r) else "irregular"
        elif r["contrast"] >= CLEAR_CONTRAST * typical and abs(r["offset_ms"]) > DRIFT_MS:
            status = "off"             # clearly locked, but beside the grid
        elif r["contrast"] < CLEAR_CONTRAST * typical and other_tempo_fits(r):
            status = "off"             # weakly locked, and another tempo fits clearly better
        else:
            status = "ok"
        info = {k: v for k, v in (r or {"offset_ms": 0.0, "contrast": 0.0, "energy": 0.0}).items()
                if k not in ("i0", "i1")}
        bars.append({"bar": b, "status": status, **info})
    return bars


def drift_summary(bars: list[dict], bpm: float, first: float, k_start: int) -> dict:
    """Find the first run of DRIFT_RUN+ consecutive 'off' bars (quiet bars in
    between don't break a run; ok or irregular bars do)."""
    beat = 60.0 / bpm
    run, start = 0, None
    for b in bars:
        if b["status"] == "off":
            if run == 0:
                start = b["bar"]
            run += 1
            if run >= DRIFT_RUN:
                t = first + (k_start + 4 * start) * beat
                return {"drift": True, "from_bar": start, "from_time": round(max(0.0, t), 3),
                        "off_bars": sum(x["status"] == "off" for x in bars),
                        "irregular_bars": sum(x["status"] == "irregular" for x in bars)}
        elif b["status"] in ("ok", "irregular"):
            run = 0
    return {"drift": False, "off_bars": sum(x["status"] == "off" for x in bars),
            "irregular_bars": sum(x["status"] == "irregular" for x in bars)}


def analyze(y: np.ndarray, sr: int, bpm: float, first: float) -> dict:
    duration = len(y) / sr
    curves, fps = stem_curves(y, sr)
    g = grid_onsets(curves, fps, bpm, first, duration)
    bars = bar_fit(curves, fps, bpm, first, g["k_start"], g["n_beats"])
    from backend.audio.structure import analyze_structure
    struct = analyze_structure(y, sr, bpm, first, g["k_start"], g["n_beats"], g["stems"])
    return {
        "version": VERSION, "bpm": bpm, "first": first,
        "k_start": g["k_start"], "n_beats": g["n_beats"], "subdiv": SUBDIV,
        # 0..255 per grid point (value 0..2 scaled), compact for JSON
        "stems": {k: np.round(v / 2 * 255).astype(int).tolist() for k, v in g["stems"].items()},
        "bars": bars,
        "summary": drift_summary(bars, bpm, first, g["k_start"]),
        **struct,                    # energy, sections, demo
    }