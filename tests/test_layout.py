"""Layout generator: playable, follows the song, calm parts calmer."""
from __future__ import annotations

import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chartcodec.notes import Hold, Slide, Tap                       # noqa: E402
from backend.chartcodec.parse import parse_chart, parse_maidata            # noqa: E402
from backend.checker.playability import check                              # noqa: E402
from backend.layout import DENSITIES, load_tables, maidata_text, make_layout  # noqa: E402
from backend.layout import rhythm                                          # noqa: E402
from backend.layout.hands import Hand, forced_hand, max_move, mirror      # noqa: E402
from backend.layout.validate import body_of, make_safe                    # noqa: E402
from test_layout_features import fake_mel                                  # noqa: E402

BPM, FIRST, DUR = 120.0, 1.5, 96.0          # the fake song hits every 8th note at 120 BPM
CALM = (40.0, 56.0)


@pytest.fixture(scope="module")
def mel():
    return fake_mel(DUR, calm=CALM, seed=3)


def _bars(notes, bpm=BPM, first=FIRST):
    bar = 4 * 60 / bpm
    c = Counter(int((first + float(n.beat) * 60 / bpm) // bar) for n in {n.beat: n for n in notes}.values())
    return c


@pytest.mark.parametrize("density", DENSITIES)
def test_every_density_is_playable(mel, density):
    lay = make_layout(mel, BPM, FIRST, DUR, density, seed=1)
    assert lay.notes
    assert check(lay.body, 0.0) == []
    assert lay.stats["checker_fixes"] == 0                   # the hand rules alone were enough
    per_beat = Counter(n.beat for n in lay.notes)
    assert max(per_beat.values()) <= 2
    buttons = Counter((n.beat, n.start if isinstance(n, Slide) else n.button) for n in lay.notes)
    assert max(buttons.values()) == 1
    for n in lay.notes:                                       # simple grid only
        assert (n.beat * 4).denominator == 1


def test_denser_settings_give_more_notes(mel):
    counts = [len(make_layout(mel, BPM, FIRST, DUR, d).notes) for d in DENSITIES]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_calm_part_gets_fewer_notes_despite_sharp_peaks(mel):
    lay = make_layout(mel, BPM, FIRST, DUR, "dense")
    bar = 4 * 60 / BPM
    per = _bars(lay.notes)
    calm = [per.get(k, 0) for k in range(int(CALM[0] / bar) + 1, int(CALM[1] / bar) - 1)]
    loud = [per.get(k, 0) for k in range(int(10 / bar), int(CALM[0] / bar) - 1)]
    assert np.mean(calm) < 0.7 * np.mean(loud)


def test_notes_sit_on_hits(mel):
    lay = make_layout(mel, BPM, FIRST, DUR, "medium")
    assert lay.stats["on_hits"] >= 0.8


def test_same_seed_same_layout(mel):
    a = make_layout(mel, BPM, FIRST, DUR, "medium", seed=5).body
    b = make_layout(mel, BPM, FIRST, DUR, "medium", seed=5).body
    c = make_layout(mel, BPM, FIRST, DUR, "medium", seed=6).body
    assert a == b and a != c


def test_hand_rules():
    h = Hand("R")
    assert h.can(1, 0.0) and not h.can(5, 0.0)               # right hand: 1-4 only
    h.play(1, 0.0, 0.0)
    assert h.can(2, 0.11) and not h.can(3, 0.11)              # reach grows with time
    assert not h.can(1, 0.1) and h.can(1, 0.15)               # jack needs ONE_HAND_SEC
    h.play(2, 1.0, 2.0)                                      # holding until 2.0
    assert not h.can(3, 1.5) and h.can(3, 2.2)
    assert forced_hand(0.1, 0.0, "R") == "L" and forced_hand(0.2, 0.0, "R") is None
    assert max_move(0.0) == 0 and max_move(0.3) == 4
    assert [mirror(b) for b in (1, 4, 5, 8)] == [8, 5, 4, 1]


def test_make_safe_repairs_an_impossible_moment():
    bad = [Tap(Fraction(0), 1), Tap(Fraction(0), 3), Tap(Fraction(0), 6), Tap(Fraction(1), 2),
           Hold(Fraction(2), 1, Fraction(2)), Tap(Fraction(3), 1)]
    assert check(body_of(bad, 120.0), 0.0)
    fixed, info = make_safe(bad, 120.0)
    assert check(body_of(fixed, 120.0), 0.0) == [] and info["checker_fixes"] >= 1


def test_maidata_round_trip(mel):
    lay = make_layout(mel, BPM, FIRST, DUR, "light")
    text = maidata_text("t", "a", BPM, FIRST, {3: (lay.level, lay.notes)})
    md = parse_maidata(text)
    assert 0 <= md["first"] < 4 * 60 / BPM                   # moved back to near the start
    chart, _ = parse_chart(md["charts"][3]["body"], None)
    shift = (FIRST - md["first"]) / (60 / BPM)
    back = sorted(float(n.beat) - shift for n in chart.notes)
    assert np.allclose(back, sorted(float(n.beat) for n in lay.notes))


def test_tables_have_every_density():
    t = load_tables()
    for d in DENSITIES:
        assert d in t["model"] and d in t["flow"] and d in t["slots"]
    assert set(rhythm.LEVEL_OF) == set(DENSITIES)


@pytest.mark.parametrize("density", ["light", "dense"])
def test_holds_and_slides_with_the_other_hand_playing(mel, density):
    lay = make_layout(mel, BPM, FIRST, DUR, density, seed=2)
    long = [n for n in lay.notes if isinstance(n, (Hold, Slide))]
    assert len(long) >= 0.03 * len(lay.notes)                # not just taps
    assert lay.stats["dropped"] == 0 and lay.stats["checker_fixes"] == 0
    side = lambda b: b <= 4                                   # noqa: E731
    for n in long:
        b = n.start if isinstance(n, Slide) else n.button
        end = n.beat + (1 + n.paths[0].segments[0].length if isinstance(n, Slide) else n.length)
        under = [x for x in lay.notes if n.beat < x.beat < end]
        assert len({x.beat for x in under}) == len(under)    # only single notes under it
        for x in under:
            xb = x.start if isinstance(x, Slide) else x.button
            assert side(xb) != side(b)                       # played by the other hand
