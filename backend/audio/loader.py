"""Audio import: track.mp3 for the game, analysis.wav for us, tags, cover art.

Important: an uploaded .mp3 is copied byte-for-byte, never re-encoded.
Re-encoding an mp3 shifts its timing by tens of ms (encoder delay/padding),
which would break any offset the user already knows (e.g. from an existing
chart).
"""
from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend import config


class AudioError(RuntimeError):
    pass


def _run_ffmpeg(args: list[str]) -> None:
    cmd = [config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", *args]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise AudioError(
            "ffmpeg not found. Install it and add it to PATH, or set the FFMPEG "
            "environment variable to ffmpeg.exe") from None
    if r.returncode != 0:
        raise AudioError(f"ffmpeg failed: {r.stderr.strip()[:500]}")


# ------------------------------------------------------------------ track

def make_track_mp3(src: Path, dst: Path) -> None:
    """dst = track.mp3 for AstroDX. mp3 input is copied untouched."""
    if src.suffix.lower() == ".mp3":
        shutil.copyfile(src, dst)
    else:
        _run_ffmpeg(["-i", str(src), "-vn", "-map", "0:a:0",
                     "-codec:a", "libmp3lame", "-q:a", "2", str(dst)])


def make_analysis_wav(track_mp3: Path, dst: Path) -> None:
    """Decode the *final* track.mp3 (what the game will play) to mono 16-bit
    wav, so analysis timing matches the exported file."""
    _run_ffmpeg(["-i", str(track_mp3), "-vn", "-ac", "1",
                 "-ar", str(config.ANALYSIS_SR), "-c:a", "pcm_s16le", str(dst)])


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Return (samples as float32 in [-1, 1], sample_rate)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise AudioError("analysis.wav must be mono 16-bit")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return data.astype(np.float32) / 32768.0, sr


# ------------------------------------------------------------------ tags

@dataclass
class Tags:
    title: str = ""
    artist: str = ""


def read_tags(path: Path) -> Tags:
    try:
        import mutagen
        f = mutagen.File(str(path), easy=True)
    except Exception:
        return Tags()
    if not f or not f.tags:
        return Tags()

    def first(key: str) -> str:
        v = f.tags.get(key)
        return str(v[0]).strip() if v else ""

    return Tags(title=first("title"), artist=first("artist"))


def extract_cover(path: Path) -> tuple[bytes, str] | None:
    """Return (image_bytes, ext) for embedded cover art, or None.
    Handles mp3 (ID3 APIC), flac/ogg pictures and m4a covr."""
    try:
        import mutagen
        f = mutagen.File(str(path))
    except Exception:
        return None
    if f is None:
        return None

    candidates: list[tuple[bytes, str]] = []
    tags = getattr(f, "tags", None)
    if tags is not None:
        # ID3: prefer front cover (type 3)
        apics = [t for k, t in tags.items() if k.startswith("APIC")] if hasattr(tags, "items") else []
        apics.sort(key=lambda a: 0 if getattr(a, "type", 0) == 3 else 1)
        candidates += [(a.data, a.mime) for a in apics]
        # MP4
        covr = tags.get("covr") if hasattr(tags, "get") else None
        if covr:
            from mutagen.mp4 import MP4Cover
            c = covr[0]
            candidates.append((bytes(c), "image/png" if c.imageformat == MP4Cover.FORMAT_PNG else "image/jpeg"))
    for p in getattr(f, "pictures", []) or []:  # FLAC
        candidates.append((p.data, p.mime))

    for data, mime in candidates:
        if not data:
            continue
        if "png" in (mime or "") or data[:8] == b"\x89PNG\r\n\x1a\n":
            return data, ".png"
        return data, ".jpg"
    return None


def save_jacket(src: Path | None, cover: tuple[bytes, str] | None, out_dir: Path) -> str | None:
    """Write bg.jpg / bg.png into out_dir. An uploaded image wins over embedded
    art. Other formats (webp, bmp) are converted to png. Returns the filename."""
    if src is not None:
        ext = src.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            shutil.copyfile(src, out_dir / "bg.jpg")
            return "bg.jpg"
        if ext == ".png":
            shutil.copyfile(src, out_dir / "bg.png")
            return "bg.png"
        _run_ffmpeg(["-i", str(src), "-frames:v", "1", str(out_dir / "bg.png")])
        return "bg.png"
    if cover is not None:
        data, ext = cover
        name = "bg" + ext
        (out_dir / name).write_bytes(data)
        return name
    return None