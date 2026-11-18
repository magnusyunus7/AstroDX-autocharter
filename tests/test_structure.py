"""Phase 2.2: energy and sections on synthetic songs with a known section plan."""
import shutil

import pytest

from backend import config
from backend.audio import features, loader

pytestmark = pytest.mark.skipif(shutil.which(config.FFMPEG) is None, reason="ffmpeg missing")


def analyze_plan(tmp_path, plan, bpm=160, first=1.0, seed=50, transitions=False):
    from tools.make_test_song import make_structured, write_mp3
    y44, truth = make_structured(bpm, first, plan, seed=seed, transitions=transitions)
    mp3, wav = tmp_path / "s.mp3", tmp_path / "s.wav"
    write_mp3(y44, mp3)
    loader.make_analysis_wav(mp3, wav)
    y, sr = loader.read_wav(wav)
    a = features.analyze(y, sr, bpm, first)
    shift = -a["k_start"] // 4           # song bar i == analysis bar i + shift
    return a, truth, shift


def test_sections_labels_energy_and_demo(tmp_path):
    plan = [("intro", 8), ("verse", 8), ("chorus", 8), ("break", 4), ("verse", 8), ("chorus", 8)]
    a, truth, shift = analyze_plan(tmp_path, plan)
    found = {s["start_bar"] - shift for s in a["sections"]}
    assert set(truth[1:-1]) <= found                         # every real boundary found
    by_start = {s["start_bar"] - shift: s for s in a["sections"]}
    verse1, chorus1, verse2, chorus2 = (by_start[b] for b in (8, 16, 28, 36))
    assert verse1["label"] == verse2["label"]                # repeats share a letter
    assert chorus1["label"] == chorus2["label"]
    intro = by_start[min(by_start)]                          # starts at the pickup bar
    assert len({verse1["label"], chorus1["label"], intro["label"]}) == 3
    assert chorus1["energy"] > verse1["energy"] > by_start[24]["energy"]   # chorus > verse > break
    assert len(a["energy"]) == a["n_beats"] // 4
    assert a["demo"]["seek"] == pytest.approx(1.0 + 16 * 4 * 60 / 160, abs=0.05)  # first chorus


def test_odd_length_sections(tmp_path):
    plan = [("intro", 6), ("verse", 10), ("chorus", 12), ("break", 6), ("chorus", 8)]
    a, truth, shift = analyze_plan(tmp_path, plan, seed=52)
    found = {s["start_bar"] - shift for s in a["sections"]}
    assert set(truth[1:-1]) <= found


def test_fill_bars_dont_pull_boundaries_one_bar_early(tmp_path):
    # real-song transitions: a snare-roll + riser bar before each chorus, and an
    # impact on each section's first beat. Without phrase snapping every chorus
    # boundary was found one bar early (on the fill).
    plan = [("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("break", 8), ("chorus", 8)]
    a, truth, shift = analyze_plan(tmp_path, plan, seed=62, transitions=True)
    found = {s["start_bar"] - shift for s in a["sections"]}
    assert set(truth[1:-1]) <= found