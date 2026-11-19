"""Generator v1 on a synthetic song: density near target, legal and playable."""
import re
import shutil

import pytest

from backend import config
from backend.audio import features, loader
from backend.generator.difficulty import targets
from backend.generator.pipeline import generate
from backend.simai.reader import note_groups, read_maidata

pytestmark = pytest.mark.skipif(shutil.which(config.FFMPEG) is None, reason="ffmpeg missing")


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    from tools.make_test_song import make_structured, write_mp3
    tmp = tmp_path_factory.mktemp("gen")
    plan = [("intro", 8), ("verse", 16), ("chorus", 16), ("break", 8), ("chorus", 16)]
    y44, _ = make_structured(170, 1.0, plan, seed=90, transitions=True, variation=True)
    write_mp3(y44, tmp / "s.mp3")
    loader.make_analysis_wav(tmp / "s.mp3", tmp / "s.wav")
    y, sr = loader.read_wav(tmp / "s.wav")
    return features.analyze(y, sr, 170, 1.0), len(y) / sr


@pytest.mark.parametrize("level", ["12", "14", "14+"])
def test_density_near_target_and_playable(analysis, level):
    a, dur = analysis
    r = generate(a, level=level, slot=5, seed=3, title="t", duration=dur)
    s, t = r["stats"], targets(level)
    assert 0.85 * t.nps <= s["nps"] <= 1.2 * t.nps
    assert s["longest_stream"] <= max(6, round(t.longest_stream * 1.3))

    md = read_maidata(r["maidata"])
    groups = note_groups(md["charts"][5], md["first"])
    assert sum(len(n) for _, n in groups) == s["notes"]
    prev = None
    for tm, notes in groups:
        btns = [int(n[0]) for n in notes]
        assert all(1 <= b <= 8 for b in btns) and len(set(btns)) == len(btns)
        if prev and len(notes) == 1 and len(prev[1]) == 1 and tm - prev[0] < 0.2:
            assert (int(prev[1][0][0]) <= 4) != (btns[0] <= 4)      # fast notes alternate hands
        prev = (tm, notes)
    assert not re.search(r"\{(24|48|96)\}", r["maidata"])            # one feel per beat


def test_same_seed_same_chart_and_reroll_differs(analysis):
    a, dur = analysis
    r1 = generate(a, level="14", seed=5, duration=dur)
    r2 = generate(a, level="14", seed=5, duration=dur)
    r3 = generate(a, level="14", seed=6, duration=dur)
    assert r1["maidata"] == r2["maidata"] and r1["maidata"] != r3["maidata"]


def test_quiet_sections_get_fewer_notes(analysis):
    a, dur = analysis
    r = generate(a, level="14", seed=1, duration=dur)
    beat = 60 / 170
    def per_sec(t0, t1):
        return sum(len(n["buttons"]) for n in r["notes"] if t0 <= n["t"] < t1) / (t1 - t0)
    chorus = per_sec(1.0 + 24 * 4 * beat, 1.0 + 40 * 4 * beat)
    brk = per_sec(1.0 + 40 * 4 * beat, 1.0 + 48 * 4 * beat)
    assert chorus > 1.5 * brk