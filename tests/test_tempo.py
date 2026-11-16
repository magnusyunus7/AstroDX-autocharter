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
    # bias is calibrated on real charts (+-4.5 ms vs these very sharp synthetic drums)
    assert abs(phase_err * beat) < 0.008
    assert round((r.first - first) / beat) % 4 == 0           # right downbeat
    assert any(c.relation == "x2" for c in r.candidates) or bpm * 2 > tempo.BPM_MAX


def test_octave_chain_rule():
    # numbers from a real chart (7 Wonders): 336 is fully hit too, but 168 is the chart BPM
    ratios = {84.0: {"body": .73, "low": .85}, 112.0: {"body": .83, "low": .98},
              168.0: {"body": .96, "low": .97}, 224.0: {"body": .68, "low": .69},
              336.0: {"body": 1.0, "low": 1.0}}
    assert tempo._choose_octave(ratios) == 168.0   # never x2/3 (224), prefer <= 330
    # Restricted Access: 370 fails the kick/bass curve
    ratios = {92.5: {"body": .78, "low": 1.0}, 185.0: {"body": 1.0, "low": .93},
              370.0: {"body": .82, "low": .38}, 123.3333: {"body": .68, "low": .35}}
    assert tempo._choose_octave(ratios) == 185.0
    # RONDO: 314 is under the cap and fully hit
    ratios = {78.5: {"body": .84, "low": .97}, 157.0: {"body": .94, "low": 1.0},
              314.0: {"body": 1.0, "low": .85}}
    assert tempo._choose_octave(ratios) == 314.0