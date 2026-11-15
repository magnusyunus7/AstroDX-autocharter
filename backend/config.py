"""Paths and external tool locations. Override with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CHARTGEN_DATA", ROOT / "data"))
PROJECTS_DIR = DATA_DIR / "projects"
EXPORTS_DIR = DATA_DIR / "exports"

# Set these if ffmpeg isn't on PATH, e.g. FFMPEG=C:\ffmpeg\bin\ffmpeg.exe
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

# Analysis audio: mono, this sample rate, 16-bit wav (decoded once by ffmpeg)
ANALYSIS_SR = 22050

# Waveform peaks: one (min, max) pair per this many analysis samples.
# 64 @ 22050 Hz = ~345 pairs per second, enough for MajdataEdit-level zoom.
PEAKS_SPP = 64

AUDIO_EXTS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}