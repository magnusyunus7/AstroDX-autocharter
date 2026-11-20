"""Generator v1 on a synthetic song: density near target, legal and playable."""
import re
import shutil

import pytest

from backend import config
from backend.audio import features, loader
from backend.checker.playability import check
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
    for tm, notes in groups:
        btns = [int(n[0]) for n in notes if n[0].isdigit()]
        assert all(1 <= b <= 8 for b in btns) and len(set(btns)) == len(btns)
    assert check(md["charts"][5], md["first"]) == []                 # two hands can play it
    # one feel per beat (straight or triplet, not both); bursts are the exception
    times = sorted({n["t"] for n in r["notes"]})
    burst = {x for x, y in zip(times, times[1:]) if y - x <= 0.067} | \
            {y for x, y in zip(times, times[1:]) if y - x <= 0.067}
    beat = 60.0 / a["bpm"]
    feel = {}
    for tm in times:
        if tm in burst:
            continue
        steps = round((tm - a["first"]) / beat * 12)
        k, p = divmod(steps, 12)
        if p % 3 and p % 4:                              # neither straight nor triplet
            raise AssertionError(f"odd position at {tm}")
        f = "s" if p % 3 == 0 and p else "t" if p % 4 == 0 and p else None
        if f:
            assert feel.setdefault(k, f) == f, f"mixed feel in beat at {tm}"


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


def test_note_types_near_targets(analysis):
    a, dur = analysis
    t = targets("14")
    runs = [generate(a, level="14", seed=seed, duration=dur) for seed in (1, 2, 3)]
    avg = {k: sum(r["stats"][k] for r in runs) / len(runs)
           for k in ("hold_pct", "slide_pm", "touch_pct", "break_pct", "each_pct")}
    assert 0.5 * t.hold_pct <= avg["hold_pct"] <= 1.6 * t.hold_pct
    assert 0.5 * t.slides_per_min <= avg["slide_pm"] <= 1.4 * t.slides_per_min
    assert 0.5 * t.touch_pct <= avg["touch_pct"] <= 1.6 * t.touch_pct
    assert 0.5 * t.break_pct <= avg["break_pct"] <= 1.6 * t.break_pct
    assert 0.8 * t.each_pct <= avg["each_pct"] <= 1.2 * t.each_pct
    for r in runs:
        kinds = {k for n in r["notes"] for k in n["kinds"]}
        assert kinds == {"tap", "hold", "slide", "touch"}
        assert all(n["t"] <= n["end"] <= dur for n in r["notes"])    # nothing runs past the audio


def test_playability_checker_catches_problems():
    assert check("(120){4}1,2/7,3h[4:1],6,E") == []
    probs = check("(120){4}1h[1:1],5,6/7,1,E")
    assert any("free hand" in p for p in probs) and any("held button" in p for p in probs)
    assert any("one hand" in p for p in check("(120){4}1h[1:1],{16}5,6,E"))
    # slide finish room: EACH / a note on or next to the slide needs a {4} after it ends
    probs = check("(170){4}2/7,1b<6[16:3]/8b>3[16:3],,3/6,E")
    assert any("EACH" in p for p in probs) and any("next to slide" in p for p in probs)
    assert check("(170){4}2/7,1b<6[8:1]/8b>3[8:1],,,3/6,E") == []
    assert any("next to slide" in p for p in check("(200){4}1-5[8:1],,{8},5,E"))  # fast BPM: more than a {4}
    assert check("(170){4}1-5[8:1],,{8},3,E") == []                           # away from the slide: fine
    assert any("next to slide 1>4" in p for p in check("(170){8}1>4[8:1],,,6,3,{4},,E"))
    assert any("touch" in p for p in check("(170){16}1-5[4:1],,,,,B5,{4},,E"))  # touch while a slide moves
    # the slide hand's next note right after the slide: from a {8} after its end,
    # right next to it (not on the end / its start / elsewhere). "Destined Marionette":
    dm = "(170){16}4/5-7[8:1],,4,,3,,2,,8,3,1,2,8,,5/6,,{16}4-2[8:1],,5,,6,,7,,1,6,8,7,1,,3/4,,{1},,E"
    assert check(dm) == []
    pre = "(170){16}4/5-7[8:1],,4,,3,,2,"
    assert any("needs" in p for p in check(pre + "8,,3,1,2,8,,,,{1},,E"))        # 8 only a {16} after
    assert any("next to its end" in p for p in check(pre + ",1,2,,,,,,,{1},,E"))  # other hand busy: 2 is the slide hand's
    assert any("next to slide" in p for p in check(pre + ",5,,,,,,,,{1},,E"))     # back on its start
    assert any("next to slide" in p for p in check(pre + ",7,,,,,,,,{1},,E"))     # on its end
    assert any("next to slide 1>4" in p for p in check("(170){8}1>4[8:1],,,6,3,{4},,E"))  # not back on the path
    # notes as a slide starts moving: one may be its own start button (tap + slide)
    assert any("free hand" in p for p in check("(170){4}1-4[8:1],2/3,{1},,E"))
    assert check("(170){4}1-4[8:1],1/8,{1},,E") == []
    assert check("(170){8}1-5[8:1],2/8,1,{1},,E") == []
    # straight / curve slides cover at least 2 buttons
    assert any("at least 2" in p for p in check("(170){4}1-2[8:1],,,,E"))
    assert any("at least 2" in p for p in check("(170){4}1<8[8:1],,,,E"))
    assert check("(170){4}1-3[8:1],,,,E") == [] and check("(170){4}1<7[8:1],,,,E") == []
    # a held button gets the same room after release
    assert any("after a hold" in p for p in check("(170){16}6h[4:1],1,,,6,{4},,E"))


def test_slide_lengths_follow_charting_habits(analysis):
    a, dur = analysis
    r = generate(a, level="14", seed=4, duration=dur)
    lengths = re.findall(r"\d[-<>^vpqw]+\d\[(\d+:\d+)\]", r["maidata"])
    assert lengths and lengths.count("8:1") / len(lengths) > 0.5           # mostly [8:1] at 170 BPM
    assert not {"16:1", "16:3", "32:1"} & set(lengths)                       # never faster than [8:1]


def test_bursts(analysis):
    a, dur = analysis
    r = generate(a, level="14", seed=1, duration=dur)
    bursts = re.findall(r"\{(24|32|48|64)\}", r["maidata"])
    assert 1 <= len(bursts) <= 8                                     # a few per song at 14, not a must
    low = generate(a, level="12", seed=1, duration=dur)                # none below 13+
    assert not re.findall(r"\{(24|32|48|64)\}", low["maidata"])