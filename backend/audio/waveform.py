"""Waveform peaks for the music bar.

Stored as peaks.bin: int8 pairs [min0, max0, min1, max1, ...], one pair per
PEAKS_SPP analysis samples, scaled so the loudest sample = 127. The frontend
fetches it once and downsamples further for zoomed-out views.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_peaks(samples: np.ndarray, spp: int) -> np.ndarray:
    """Return int8 array of shape (n_buckets, 2) with per-bucket (min, max)."""
    n = len(samples) // spp
    if n == 0:
        return np.zeros((0, 2), dtype=np.int8)
    x = samples[: n * spp].reshape(n, spp)
    mn, mx = x.min(axis=1), x.max(axis=1)
    peak = float(max(np.abs(mn).max(), np.abs(mx).max(), 1e-9))
    out = np.empty((n, 2), dtype=np.int8)
    out[:, 0] = np.clip(np.round(mn / peak * 127), -127, 127)
    out[:, 1] = np.clip(np.round(mx / peak * 127), -127, 127)
    return out


def write_peaks(peaks: np.ndarray, path: Path) -> None:
    path.write_bytes(peaks.astype(np.int8).tobytes())


def first_sound_time(samples: np.ndarray, sr: int, threshold_db: float = -40.0) -> float:
    """Seconds until the audio first gets louder than threshold (relative to
    the track's peak). Useful hint for where the music actually starts."""
    peak = float(np.abs(samples).max()) or 1.0
    thr = peak * (10 ** (threshold_db / 20))
    hop = max(1, sr // 100)  # 10 ms windows
    n = len(samples) // hop
    if n == 0:
        return 0.0
    frames = np.abs(samples[: n * hop].reshape(n, hop)).max(axis=1)
    idx = np.argmax(frames > thr)
    return float(idx * hop / sr) if frames[idx] > thr else 0.0