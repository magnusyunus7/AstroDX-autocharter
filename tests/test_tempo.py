"""Timing detection on synthetic songs with exactly known BPM/downbeat."""
import shutil

import pytest

from backend import config
from backend.audio import loader, tempo


def test_allowed_and_snap():
    bpms = tempo.allowed_bpms(150, 151)
    assert [round(b, 4) for b in bpms] == [150.0, 150.3333, 150.5, 150.6667, 151.0]
    assert tempo.snap_bpm(157.334) == 157.3333
    assert tempo.snap_bpm(200.49) == 200.5
    assert tempo.snap_bpm(150.2) is None


@pytest.mark.skipif(shutil.which(config.FFMPEG) is None, reason="ffmpeg missing")
@pytest.mark.parametrize("bpm,first", [(157.3333, 1.234), (314, 0.5), (96, 1.6)])
def test_detect_synthetic(tmp_path, bpm, first):
    from tools.make_test_song import make, write_mp3
    mp3, wav = tmp_path / "s.mp3", tmp_path / "s.wav"
    write_mp3(make(bpm, first, 32, seed=3), mp3)
    loader.make_analysis_wav(mp3, wav)
    y, sr = loader.read_wav(wav)
    r = tempo.detect(y, sr)

    assert r.bpm == pytest.approx(bpm, abs=1e-3)
    beat = 60 / bpm
    phase_err = ((r.beat_phase - first) / beat + 0.5) % 1 - 0.5
    assert abs(phase_err * beat) < 0.003                      # within 3 ms
    assert round((r.first - first) / beat) % 4 == 0           # right downbeat
    assert any(c.relation == "x2" for c in r.candidates) or bpm * 2 > tempo.BPM_MAX