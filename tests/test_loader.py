"""Loader/storage tests. Generates its own audio with ffmpeg (skipped if
ffmpeg isn't installed)."""
import shutil
import subprocess

import numpy as np
import pytest

from backend import config, storage
from backend.audio import loader, waveform

pytestmark = pytest.mark.skipif(shutil.which(config.FFMPEG) is None, reason="ffmpeg missing")


def ff(*args):
    subprocess.run([config.FFMPEG, "-loglevel", "error", "-y", *args], check=True)


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")
    return tmp_path


def tone(path, seconds=5.0, delay=0.0):
    """Sine tone, optionally preceded by `delay` seconds of silence."""
    if delay:
        ff("-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
           "-af", f"adelay={int(delay * 1000)}", str(path))
    else:
        ff("-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", str(path))
    return path


def test_mp3_copied_untouched(tmp_path):
    src = tone(tmp_path / "a.mp3")
    p = storage.create_project(src)
    d = storage.project_dir(p.id)
    assert (d / "track.mp3").read_bytes() == src.read_bytes()
    assert abs(p.duration - 5.0) < 0.1
    assert p.title == "a"
    assert (d / "peaks.bin").stat().st_size > 0
    assert not list(d.glob("source*"))  # original upload isn't duplicated


def test_wav_converted_and_tags(tmp_path):
    src = tmp_path / "b.flac"
    ff("-f", "lavfi", "-i", "sine=frequency=440:duration=3",
       "-metadata", "title=My Song", "-metadata", "artist=Someone", str(src))
    p = storage.create_project(src)
    d = storage.project_dir(p.id)
    assert (d / "track.mp3").read_bytes()[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")
    assert p.title == "My Song" and p.artist == "Someone"


def test_embedded_cover_extracted(tmp_path):
    from mutagen.id3 import APIC, ID3
    src = tone(tmp_path / "c.mp3")
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    tags = ID3()
    tags.add(APIC(encoding=3, mime="image/png", type=3, desc="cover", data=png))
    tags.save(str(src))
    p = storage.create_project(src)
    assert p.jacket == "bg.png"
    assert (storage.project_dir(p.id) / "bg.png").read_bytes() == png


def test_uploaded_jacket_wins(tmp_path):
    src = tone(tmp_path / "d.mp3")
    jpg = tmp_path / "j.jpg"
    jpg.write_bytes(b"\xff\xd8\xff fake jpeg")
    p = storage.create_project(src, jacket_path=jpg)
    assert p.jacket == "bg.jpg"


def test_first_sound_detects_silence(tmp_path):
    src = tone(tmp_path / "e.wav", seconds=3, delay=1.5)
    p = storage.create_project(src)
    samples, sr = loader.read_wav(storage.project_dir(p.id) / "analysis.wav")
    t = waveform.first_sound_time(samples, sr)
    # mp3 encoder delay adds a few tens of ms
    assert 1.45 < t < 1.6


def test_peaks_shape():
    x = np.sin(np.linspace(0, 100, 64 * 10)).astype(np.float32)
    pk = waveform.compute_peaks(x, 64)
    assert pk.shape == (10, 2) and pk.dtype == np.int8
    assert pk[:, 1].max() == 127


def test_bad_inputs(tmp_path):
    txt = tmp_path / "x.txt"
    txt.write_text("nope")
    with pytest.raises(ValueError):
        storage.create_project(txt)
    with pytest.raises(ValueError):
        storage.project_dir("../evil")
    fake = tmp_path / "broken.mp3"
    fake.write_bytes(b"not audio")
    with pytest.raises(loader.AudioError):
        storage.create_project(fake)
    assert not any((tmp_path / "projects").iterdir())  # failed import cleaned up


def test_title_from_sibling_maidata_or_folder(tmp_path):
    lvl = tmp_path / "RONDØ"
    lvl.mkdir()
    src = tone(lvl / "track.mp3")
    p = storage.create_project(src)
    assert p.title == "RONDØ" and p.artist == ""          # folder name
    (lvl / "maidata.txt").write_text("&title=RONDØ\n&artist=Ashrount\n&first=0\n", encoding="utf-8")
    p = storage.create_project(src)
    assert p.title == "RONDØ" and p.artist == "Ashrount"  # from maidata.txt


def test_analysis_wav_is_rebuildable(tmp_path):
    p = storage.create_project(tone(tmp_path / "r.mp3"))
    wav = storage.project_dir(p.id) / "analysis.wav"
    before = wav.read_bytes()
    wav.unlink()
    assert storage.ensure_analysis_wav(p.id).read_bytes() == before