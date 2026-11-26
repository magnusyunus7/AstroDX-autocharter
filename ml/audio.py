"""Audio for the chart model (Phase C2).

Decoding goes through ffmpeg exactly like the app does (backend/audio/loader:
mono, 22050 Hz), so the model hears training songs and new songs with the
same timing; mp3 decoders differ by tens of ms otherwise.

Features: a log-mel spectrogram at 100 frames per second (frame k covers the
moment k / FPS seconds). MERT features are made separately (ml/mert_features.py).
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

SR = 22050                     # same as the app (backend/config.ANALYSIS_SR)
HOP = 220                      # 10 ms
FPS = SR / HOP                 # 100.227 frames per second
N_MELS = 80
N_FFT = 2048


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found (install it, or pip install imageio-ffmpeg)") from None


def decode(path, sr: int = SR) -> np.ndarray:
    """Audio file -> mono float32 samples at sr, the way the app decodes it."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr.decode(errors='replace')[:300]}")
    return np.frombuffer(r.stdout, dtype="<f4").copy()


def mel(y: np.ndarray) -> np.ndarray:
    """(frames, N_MELS) log-mel in dB (about -80..0), float16. Frame k is
    centred on sample k * HOP."""
    import librosa
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
                                       fmin=30.0, fmax=SR / 2, power=2.0)
    db = librosa.power_to_db(S, ref=np.max, top_db=80.0)
    return db.T.astype(np.float16)
