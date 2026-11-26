"""Sync check (Phase C2): does a chart line up with its audio?

For every chart: slide the whole chart earlier / later by up to SEARCH ms and
find where its notes sit on the strongest hits in the audio (the app's own
onset curve, backend/audio/tempo.py). The best shift is the chart's offset:
about 0 for a chart that's in sync. Charts whose offset is far from the
collection's usual value are out of sync with their track (wrong &first, a
different edit of the song ...).

sharpness says how clear the answer is: a dense chart over drums gives a
tall narrow peak (high), a sparse chart over pads gives a flat curve (low,
offset can't be trusted).

The search stops short of half a beat: a sparse chart (quarter notes) over a
song with strong off-beat hits would otherwise "line up" best half a beat
away. An answer at the edge of the search (edge=True) isn't trusted either.
"""
from __future__ import annotations

import numpy as np

SEARCH = 0.150                 # seconds each way


def onset_envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """The app's attack curve (~2.9 ms frames), normalised."""
    from backend.audio import tempo
    attack, _, _, _, fps = tempo.onset_curves(y, sr, separate=False)
    env = attack.astype(np.float64)
    env = (env - np.median(env)) / (env.std() + 1e-9)
    return env, fps


def chart_offset(env: np.ndarray, fps: float, times: np.ndarray, beat: float | None = None) -> dict:
    """times: note moments in seconds from the audio start. -> {'offset_ms',
    'sharpness', 'gain'}: the shift that puts the notes on the hits (positive:
    the audio's hits come later than the chart's notes), how clear that is,
    and how much better it is than no shift at all."""
    reach = min(SEARCH, 0.4 * beat) if beat else SEARCH
    times = np.asarray(times, dtype=np.float64)
    times = times[(times > reach) & (times < len(env) / fps - reach)]
    if len(times) < 20:
        return {"offset_ms": None, "sharpness": 0.0, "gain": 0.0, "edge": False}
    shifts = np.arange(-int(reach * fps), int(reach * fps) + 1)
    base = np.round(times * fps).astype(np.int64)
    score = np.array([env[base + s].mean() for s in shifts])
    k = int(np.argmax(score))
    sharp = (score[k] - np.median(score)) / (score.std() + 1e-9)
    zero = score[len(shifts) // 2]
    return {"offset_ms": round(float(shifts[k] / fps * 1000), 1), "sharpness": round(float(sharp), 2),
            "gain": round(float(score[k] - zero), 3), "edge": k in (0, len(shifts) - 1)}
