"""Audio measurements for the layout generator, numpy only (runs on device).

Everything starts from the log-mel spectrogram the app and the dataset
already make (ml/audio.mel: dB, about -80..0 relative to the song's loudest
frame, 80 bands, 100.227 frames per second).

Two kinds of measurement:

  bar ratings   how busy each bar of the song is *compared to the rest of the
                same song*: loudness and hit strength as a percentile (0 =
                the song's calmest bar, 1 = its busiest). A quiet verse ranks
                low even when its piano has sharp peaks: that is what lets
                calm parts get few notes, like real charts.
  slot strength how strong the hit is at each 16th-note slot, also as a
                percentile within the song, plus the slot's place in the bar
                (downbeat / beat / 8th / 16th).

Bins (used by the tables): 5 loudness bins x 5 hit-strength bins per bar,
5 strength bins per slot, 4 metric classes.
"""
from __future__ import annotations

import numpy as np

FPS = 22050 / 220                 # mel frames per second (ml/audio.FPS)
LOW_BANDS = 12                    # about 30-250 Hz: kick / bass hits
SLOT_WIN = 2                      # frames either side when reading a slot's hit (about 20 ms)
BINS = 5
METRIC = ("downbeat", "beat", "8th", "16th")


def curves(mel_db: np.ndarray) -> dict[str, np.ndarray]:
    """Frame curves from a (frames, bands) dB spectrogram:
    loud  mean level (dB)
    flux  hit strength: how much the spectrum rises from the frame before
    low   the same, low bands only (drums / bass)"""
    m = np.asarray(mel_db, dtype=np.float32)
    d = np.maximum(np.diff(m, axis=0, prepend=m[:1]), 0.0)
    return {"loud": m.mean(1), "flux": d.mean(1), "low": d[:, :LOW_BANDS].mean(1)}


def percentile(x: np.ndarray) -> np.ndarray:
    """Each value's rank within x, 0..1 (ties share their average rank)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x
    if n == 1:
        return np.full(1, 0.5)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    # average ranks of ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv] / (n - 1)


def to_bin(p: np.ndarray, bins: int = BINS) -> np.ndarray:
    return np.minimum(bins - 1, (np.asarray(p) * bins).astype(np.int64))


def _span_mean(c: np.ndarray, f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(c, dtype=np.float64)])
    f0 = np.clip(f0, 0, len(c)); f1 = np.clip(np.maximum(f1, f0 + 1), 0, len(c))
    n = np.maximum(1, f1 - f0)
    return (cs[f1] - cs[f0]) / n


def bar_ratings(cv: dict, t0: np.ndarray, t1: np.ndarray) -> dict[str, np.ndarray]:
    """Per bar (start / end seconds): loudness and hit-strength percentiles
    within the given bars, and their bins."""
    f0 = np.floor(np.asarray(t0) * FPS).astype(np.int64)
    f1 = np.ceil(np.asarray(t1) * FPS).astype(np.int64)
    loud = _span_mean(cv["loud"], f0, f1)
    hit = _span_mean(cv["flux"], f0, f1) + _span_mean(cv["low"], f0, f1)
    lp, hp = percentile(loud), percentile(hit)
    return {"loud": lp, "hit": hp, "loud_bin": to_bin(lp), "hit_bin": to_bin(hp)}


def slot_strength(cv: dict, times: np.ndarray) -> np.ndarray:
    """Hit strength at each time (seconds): the strongest flux (all bands plus
    low bands) within about 20 ms."""
    c = cv["flux"] + cv["low"]
    f = np.round(np.asarray(times) * FPS).astype(np.int64)
    out = np.zeros(len(f))
    for o in range(-SLOT_WIN, SLOT_WIN + 1):
        out = np.maximum(out, c[np.clip(f + o, 0, len(c) - 1)])
    out[(f < 0) | (f >= len(c))] = 0.0
    return out


def metric_class(slot16: np.ndarray) -> np.ndarray:
    """Place in the bar for 16th slots 0..15: 0 downbeat, 1 beat, 2 8th, 3 16th."""
    s = np.asarray(slot16) % 16
    return np.where(s == 0, 0, np.where(s % 4 == 0, 1, np.where(s % 2 == 0, 2, 3)))
