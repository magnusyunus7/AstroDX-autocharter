"""Phase 2 analysis on synthetic songs with known events."""
import shutil

import numpy as np
import pytest

from backend import config
from backend.audio import features, loader

pytestmark = pytest.mark.skipif(shutil.which(config.FFMPEG) is None, reason="ffmpeg missing")


def render(tmp_path, y44, name="s"):
    from tools.make_test_song import write_mp3
    mp3, wav = tmp_path / f"{name}.mp3", tmp_path / f"{name}.wav"
    write_mp3(y44, mp3)
    loader.make_analysis_wav(mp3, wav)
    return loader.read_wav(wav)


def test_grid_layout_starts_on_a_bar_line():
    k_start, n_beats = features.grid_layout(170, 1.0, 30.0)
    assert k_start % 4 == 0 and n_beats % 4 == 0
    beat = 60 / 170
    assert 1.0 + k_start * beat <= 0 < 1.0 + (k_start + 4) * beat


def test_stems_hit_known_events_and_no_drift(tmp_path):
    from tools.make_test_song import make
    bpm, first = 170, 1.0
    events = []
    y, sr = render(tmp_path, make(bpm, first, 20, seed=31, events=events))
    a = features.analyze(y, sr, bpm, first)
    beat, sub = 60 / bpm, a["subdiv"]

    def val(stem, t):
        i = int(round(((t - first) / beat - a["k_start"]) * sub))
        return a["stems"][stem][i]

    for kind, stem in (("kick", "drums"), ("kick", "bass"), ("chord", "melody")):
        on = np.median([val(stem, t) for t, k in events if k == kind])
        off = np.median([val(stem, t + beat / 6) for t, k in events if k == kind])
        assert on > 40 and on > 1.5 * off, (kind, stem, on, off)
    assert a["summary"]["drift"] is False
    assert a["summary"]["off_bars"] == 0
    assert "quiet" in {b["status"] for b in a["bars"]}   # the drumless breakdown


def test_tempo_change_is_flagged_where_it_happens(tmp_path):
    from tools.make_test_song import SR, make
    y1 = make(170, 1.0, 16, seed=32)
    cut = int((1.0 + 16 * 4 * 60 / 170) * SR)          # end of bar 16
    y2 = make(164, 0.0, 12, seed=33)
    y, sr = render(tmp_path, np.concatenate([y1[:cut], y2]))
    a = features.analyze(y, sr, 170, 1.0)
    s = a["summary"]
    assert s["drift"] is True
    change_t = 1.0 + 16 * 4 * 60 / 170
    assert abs(s["from_time"] - change_t) < 2 * 4 * 60 / 170   # within 2 bars