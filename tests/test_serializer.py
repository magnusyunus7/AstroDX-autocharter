from fractions import Fraction as F

import pytest

from backend.models.chart import (
    Chart, ChartError, Hold, Slide, Tap, Touch, TouchHold,
)
from backend.simai.maidata import (
    DifficultyChart, MaidataMeta, build_maidata, rebase_first,
)
from backend.simai.serializer import fmt_duration, note_token, serialize


def test_durations():
    assert fmt_duration(F(1)) == "[4:1]"
    assert fmt_duration(F(2)) == "[2:1]"
    assert fmt_duration(F(3, 2)) == "[8:3]"
    assert fmt_duration(F(1, 3)) == "[12:1]"


def test_tokens():
    assert note_token(Tap(0, 3)) == "3"
    assert note_token(Tap(0, 3, brk=True, ex=True)) == "3bx"
    assert note_token(Hold(0, 4, 2)) == "4h[2:1]"
    assert note_token(Slide(0, 1, "-", 5, 1)) == "1-5[4:1]"
    assert note_token(Slide(0, 1, ">", 1, 2, brk=True)) == "1b>1[2:1]"
    assert note_token(Touch(0, "C", firework=True)) == "Cf"
    assert note_token(TouchHold(0, 2, firework=True)) == "Chf[2:1]"


def test_quarter_bar():
    c = Chart(120).add(*(Tap(i, 1) for i in range(4)))
    assert serialize(c) == "(120){4}1,1,1,1,\nE"


def test_empty_bar_and_each():
    c = Chart(150.5).add(Tap(0, 1), Tap(0, 8), Tap(8, 2))
    assert serialize(c) == "(150.5){4}1/8,,,,\n{1},\n{4}2,,,,\nE"


def test_mixed_divisor_bar_splits_per_beat():
    # 8ths in beat 0, triplet in beat 1 -> per-beat divisors, not {24}
    c = Chart(180).add(Tap(0, 1), Tap(F(1, 2), 2), Tap(F(4, 3), 3))
    line = serialize(c).splitlines()[0]
    assert line == "(180){8}1,2,{12},3,,{4},,"


def test_16ths_stay_single_divisor():
    c = Chart(200).add(*(Tap(F(i, 4), 1 + i % 2) for i in range(16)))
    assert serialize(c).splitlines()[0] == "(200){16}" + "1,2," * 8


def test_hold_extends_bars():
    # hold starting bar 0 beat 3 lasting 6 beats -> ends beat 9 -> needs 3 bars
    c = Chart(120).add(Hold(3, 1, 6))
    assert serialize(c).count("\n") == 3  # 3 bar lines + E


def test_slide_end_includes_wait():
    s = Slide(2, 1, "-", 5, 2)
    assert s.end == 5  # 2 + 1 wait + 2 travel


def test_invalid_notes():
    with pytest.raises(ChartError):
        serialize(Chart(120).add(Tap(0, 9)))
    with pytest.raises(ChartError):
        serialize(Chart(120).add(Slide(0, 1, "-", 2, 1)))  # adjacent straight
    with pytest.raises(ChartError):
        serialize(Chart(120).add(Tap(0, 1), Tap(0, 1)))
    with pytest.raises(ChartError):
        serialize(Chart(120).add(Touch(0, "F1")))


def test_rebase_first():
    # 120 bpm -> bar = 2s. first downbeat at 9.5s -> 4 bars back -> 1.5s
    first, bars = rebase_first(9.5, 120)
    assert bars == 4 and abs(first - 1.5) < 1e-9


def test_maidata_and_length_check():
    c = Chart(120).add(Tap(0, 1))
    text = build_maidata(MaidataMeta("A & B", "x", first=0.5),
                         [DifficultyChart(5, "14+", c)])
    assert "&lv_5=14+" in text and "&inote_5=(120){4}1,,,,\nE" in text
    assert "&title=A ＆ B" in text
    with pytest.raises(ValueError):
        build_maidata(MaidataMeta("t", first=10), [DifficultyChart(5, "1", c)],
                      audio_duration=5)