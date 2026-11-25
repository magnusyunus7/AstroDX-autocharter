"""Phase C1: simai parser / writer, tokens, rotate / mirror."""
from fractions import Fraction as F

import pytest

from backend.chartcodec.augment import transform
from backend.chartcodec.notes import (ChartData, Hold, Path, Segment, Slide, Tap, Touch, TouchHold,
                                      clean, model_view, same_notes, sort_key)
from backend.chartcodec.parse import ParseError, parse_chart, parse_maidata, split_notes
from backend.chartcodec.tokens import VOCAB, TokenError, decode, encode
from backend.chartcodec.write import write_chart


def notes_of(body, **kw):
    chart, issues = parse_chart(body, **kw)
    return chart.notes


def one(body):
    n = notes_of("(120){4}" + body + ",E")
    assert len(n) == 1, n
    return n[0]


def roundtrip(body):
    """parse -> write -> parse and parse -> tokens -> notes give the same chart."""
    chart, _ = parse_chart(body)
    again, _ = parse_chart(write_chart(chart))
    assert again.tempo == chart.tempo and same_notes(again.notes, chart.notes)
    view, _ = model_view(chart.notes)
    assert decode(encode(view)) == view
    return chart


# ---------------------------------------------------------------- timing

def test_commas_divisors_and_bpm_changes():
    c, _ = parse_chart("(120){4}1,2,{8}3,4,(240){4}5,,6,E")
    assert [(n.beat, n.button) for n in c.notes] == [
        (0, 1), (1, 2), (2, 3), (F(5, 2), 4), (3, 5), (5, 6)]
    assert c.tempo == [(0, 120.0), (3, 240.0)]
    assert c.seconds(F(3)) == pytest.approx(1.5)          # 3 beats at 120
    assert c.seconds(F(5)) == pytest.approx(2.0)          # + 2 beats at 240


def test_seconds_commas_are_turned_into_beats():
    c, issues = parse_chart("(120){#0.25}1,2,{4}3,E")     # 0.25 s = half a beat at 120
    assert [n.beat for n in c.notes] == [0, F(1, 2), 1]
    assert not issues.count


def test_times_stay_exact():
    c, issues = parse_chart("(120){20}1,2,3,4,5,{4}6,E")
    assert [n.beat for n in c.notes] == [0, F(1, 5), F(2, 5), F(3, 5), F(4, 5), 1]
    c, _ = parse_chart("(140){8}1,{625}2,,,,,,,,,,,,,,{622}3,,,,,,,,,,,,,,{625}4,,,,,,,,,,,,,,{8}5,E")
    assert c.notes[2].beat == F(1, 2) + F(56, 625)                     # old converter filler, exact
    assert c.notes[3].beat == F(1, 2) + F(56, 625) + F(56, 622)
    assert not clean(c.notes[2].beat) and clean(F(2, 5)) and clean(F(7, 768) * 4)


def test_model_view_moves_only_odd_values():
    c, _ = parse_chart("(140){8}1,{625}2,,,,,,,,,,,,,,{8}3,{20}4,{768}5,E")
    view, moved = model_view(c.notes)
    assert moved == 3                                      # notes 3, 4, 5 come after the filler
    assert [n.beat for n in view][:2] == [0, F(1, 2)]
    assert all(clean(n.beat) for n in view)
    assert c.notes[2].beat != view[2].beat and abs(float(c.notes[2].beat - view[2].beat)) <= 1 / 384


def test_comments_and_end_marker():
    c, _ = parse_chart("(120){4}1,|| a comment, with commas\n2,E")
    assert [n.beat for n in c.notes] == [0, 1]


def test_hs_speed_changes_are_skipped():
    assert len(notes_of("(120){4}<HS*2.0>1,2,E")) == 2


def test_wholebpm_when_chart_has_no_bpm():
    assert notes_of("{4}1,2,E", default_bpm=150)[1].beat == 1
    with pytest.raises(ParseError):
        parse_chart("{4}1,2,E")


# ---------------------------------------------------------------- notes

def test_taps_and_marks():
    assert one("1") == Tap(F(0), 1)
    assert one("3bx") == Tap(F(0), 3, brk=True, ex=True)
    assert one("4$") == Tap(F(0), 4, star=1)
    assert one("4$$") == Tap(F(0), 4, star=2)


def test_each_forms():
    assert [n.button for n in notes_of("(120){4}1/5,E")] == [1, 5]
    assert [n.button for n in notes_of("(120){4}15,E")] == [1, 5]
    n = notes_of("(120){4}87-4[8:1],E")                   # tap 8 + slide 7-4, no "/"
    assert {type(x) for x in n} == {Tap, Slide}
    assert split_notes("7<3[4:1]8p3[4:1]") == ["7<3[4:1]", "8p3[4:1]"]
    assert split_notes("1V35[4:1]2") == ["1V35[4:1]", "2"]


def test_pseudo_each_and_yellow_placeholder():
    n = notes_of("(120){4}3`1,E")
    assert [(x.button, x.sub) for x in n] == [(3, 0), (1, 1)]
    c, issues = parse_chart("(120){4}3/0,E")
    assert c.notes == [Tap(F(0), 3)] and issues.count["yellow-only 0 dropped"] == 1


def test_holds():
    assert one("1h[4:1]") == Hold(F(0), 1, F(1))
    assert one("2hbx[8:3]") == Hold(F(0), 2, F(3, 2), brk=True, ex=True)
    assert one("2bh[2:1]").brk and one("8hx[4:1]").ex
    assert one("7h[#1.5]").length == 3                     # 1.5 s at 120 = 3 beats
    assert one("5h").length == 0                           # instant hold


def test_touch():
    assert one("E2") == Touch(F(0), "E2")
    assert one("Cf") == Touch(F(0), "C", fw=True)
    assert one("C1") == Touch(F(0), "C")
    assert one("Chf[4:1]") == TouchHold(F(0), "C", F(1), fw=True)
    assert one("Cfh[4:1]") == TouchHold(F(0), "C", F(1), fw=True)
    assert one("C1xh[4:1]").ex


def test_slide_shapes_and_direction():
    s = one("1-5[8:1]")
    assert s.start == 1 and s.paths == (Path((Segment("-", 5, None, F(1, 2)),)),)
    assert one("1>3[8:1]").paths[0].segments[0].shape == "cw"     # top half: > = clockwise
    assert one("1<7[8:1]").paths[0].segments[0].shape == "ccw"
    assert one("5>3[8:1]").paths[0].segments[0].shape == "ccw"    # bottom half: > = counter-clockwise
    assert one("4<6[8:1]").paths[0].segments[0].shape == "cw"
    for shape in ("^", "v", "p", "q", "pp", "qq", "s", "z", "w"):
        assert one(f"1{shape}5[8:1]").paths[0].segments[0].shape == shape
    v = one("2bV46[4:1]")
    assert v.brk and v.paths[0].segments[0] == Segment("V", 6, 4, F(1))


def test_slide_chains_multi_paths_waits_and_heads():
    chain = one("1-4-7[4:1]").paths[0].segments
    assert [(s.end, s.length) for s in chain] == [(4, None), (7, F(1))]
    per = one("4qq4[8:1]z8[8:1]").paths[0].segments
    assert [(s.shape, s.length) for s in per] == [("qq", F(1, 2)), ("z", F(1, 2))]
    multi = one("8s4[12:8]*v3[12:8]")
    assert [p.segments[0].shape for p in multi.paths] == ["s", "v"]
    assert one("1-5[8:1]b").paths[0].brk and one("1-5b[8:1]").paths[0].brk
    assert one("1-5[240#8:1]").paths[0].wait == F(1, 2)    # 1 beat at 240 = half a beat at 120
    assert one("1-5[240#8:1]").paths[0].segments[0].length == F(1, 4)
    w = one("1-5[0.25##0.5]").paths[0]                     # wait 0.25 s, slide 0.5 s
    assert (w.wait, w.segments[0].length) == (F(1, 2), F(1))
    assert one("1-5[#0.5]").paths[0].segments[0].length == 1
    assert one("1-5[0:0]").paths[0].segments[0].length == 0
    assert [one(f"1{m}-5[8:1]").head for m in "?!@$"] == ["none", "sudden", "tap", "pad"]


@pytest.mark.parametrize("bad", ["9", "1h[4:1", "1-5", "1y", "F1", "Cq", "1?!-5[8:1]", "1-5[8:1]*",
                                 "C8", "2>6", "3[4:1]", "2hh[4:1]", "7-4x[4:3]"])
def test_bad_notes_raise(bad):
    with pytest.raises(ParseError):
        parse_chart(f"(120){{4}}{bad},E")


@pytest.mark.parametrize("typo, fixed", [
    ("3[4:1]", "3h[4:1]"), ("4x[4:1]", "4xh[4:1]"), ("Cf[1:1]", "Chf[1:1]"), ("2hh[12:1]", "2h[12:1]"), ("7xbhh[8:1]", "7bxh[8:1]"),
    ("7-4x[4:3]", "7x-4[4:3]"), ("2>5xb[4:1]", "2x>5[4:1]b"), ("8xp4[8:1]*x<4[8:1]", "8xp4[8:1]*<4[8:1]")])
def test_repair_mode_fixes_known_typos(typo, fixed):
    c, issues = parse_chart(f"(120){{4}}{typo},E", repair=True)
    assert c.notes == notes_of(f"(120){{4}}{fixed},E")
    assert sum(v for k, v in issues.count.items() if k.startswith("fixed")) == 1
    for still_bad in ("C8", "2>6"):                        # not guessable: stay errors
        with pytest.raises(ParseError):
            parse_chart(f"(120){{4}}{still_bad},E", repair=True)


# ---------------------------------------------------------------- round trips

ALL_SYNTAX = """(150){4}1,2bx,3$,4$$,
{8}1/5,26,87-4[8:1],3`1,
1h[4:1],2hbx[8:3],5h,E2/Cf,
{16}Chf[4:1],C1xh[2:1],1-5[8:1],1>3[8:1],5>3[8:1],2bV46[4:1],1-4-7[4:1],4qq4[8:1]z8[8:1],
8s4[12:8]*v3[12:8],1-5[8:1]b,1-5[180#8:1],1-5[0.2##8:1],1?-5[8:1],1!pp5[8:1],1@w5[8:1],1-5[0:0],
(200){12}1,2,3,{20}4,5,6,7,8,{4}1,,(150)2,{#0.1}3,4,{4}5,{768}6,7,{256}8,{625}1,,,{4}2,E"""


def test_everything_round_trips():
    c = roundtrip(ALL_SYNTAX)
    assert len(c.notes) > 40 and len(c.tempo) == 3


def test_written_text_is_readable_simai():
    text = write_chart(parse_chart("(120){8}1,,2-6[8:1]b,,{4}3h[4:1],E")[0])
    assert text.splitlines()[0] == "(120){4}1,2-6[8:1]b,3h[4:1],,"        # smallest {divisor} that fits
    assert text.endswith("E")


def test_maidata_fields():
    m = parse_maidata("&title=X\n&first=0.5\n&wholebpm=150\n&lv_5=14+\n&inote_5=(150){4}1,E\n&lv_4=12\n&inote_4={4}2,E\n")
    assert m["first"] == 0.5 and m["wholebpm"] == 150 and m["title"] == "X"
    assert m["charts"][5]["level"] == "14+" and m["charts"][4]["body"].strip() == "{4}2,E"


# ---------------------------------------------------------------- tokens

def test_tokens_example_and_vocab():
    toks = encode(notes_of("(120){4}3/6,{8},5-1[8:1],E"))
    assert toks == ["BOS", "BEAT", "P0", "TAP", "B3", "TAP", "B6",           # beat 0
                    "BEAT", "P96", "SLIDE", "B5", "PATH", "SH_-", "B1", "L0", "F96", "EOS"]  # beat 1.5
    assert encode(notes_of("(120){20},1h[20:3],E"))[2:8] == ["T5", "N1", "HOLD", "B1", "L0", "T5"]
    with pytest.raises(TokenError):                        # odd values must go through model_view
        encode(notes_of("(120){625},1,E"))
    assert len(VOCAB) == len(set(VOCAB))


def test_long_holds_and_decode_errors():
    n = [Hold(F(0), 1, F(70, 1))]
    assert decode(encode(n)) == n                          # L32 L32 L6
    for bad in (["BOS", "TAP", "B1"], ["BOS", "BEAT", "P0", "HOLD", "B1"],
                ["BOS", "BEAT", "P0", "SLIDE", "B1", "PATH", "SH_-", "B5"], ["BOS", "BEAT", "P0", "B1"]):
        with pytest.raises(TokenError):
            decode(bad)


# ---------------------------------------------------------------- rotate / mirror

def test_rotation_and_mirror_are_exact_relabellings():
    c, _ = parse_chart(ALL_SYNTAX)
    key = lambda ns: sorted(ns, key=sort_key)
    for r in range(8):
        assert key(transform(transform(c.notes, r), 8 - r)) == c.notes
    assert key(transform(transform(c.notes, 0, True), 0, True)) == c.notes
    for r in range(8):                                     # every variant still writes and reads back
        for m in (False, True):
            moved = ChartData(c.tempo, key(transform(c.notes, r, m)))
            again, _ = parse_chart(write_chart(moved))
            assert same_notes(again.notes, moved.notes)


def test_rotation_keeps_the_physical_direction():
    turn = lambda body, r, m=False: write_chart(ChartData([(F(0), 120.0)], transform(notes_of(body), r, m))).split(",")[0]
    assert turn("(120){4}1>3[8:1],E", 4) == "(120){1}5<7[8:1]"     # still clockwise
    assert turn("(120){4}1>3[8:1],E", 0, True) == "(120){1}8<6[8:1]"  # mirrored: counter-clockwise
    assert turn("(120){4}1p5[8:1],E", 0, True) == "(120){1}8q4[8:1]"
    t = transform([Touch(F(0), s) for s in ("A1", "B2", "D1", "D2", "E5", "C")], 0, True)
    assert [x.sensor for x in t] == ["A8", "B7", "D1", "D8", "E5", "C"]
    t = transform([Touch(F(0), s) for s in ("A8", "D8", "D1")], 1)
    assert [x.sensor for x in t] == ["A1", "D1", "D2"]


def test_chart_model_view_moves_odd_tempo_changes_too():
    c, _ = parse_chart("(120){8}1,{625}2,,,(150){4}3,4,E")
    view, moved = c.model_view()
    assert all(clean(b) for b, _ in view.tempo) and moved == 3      # tempo change + notes 3, 4
    again, _ = parse_chart(write_chart(view))
    assert again.tempo == view.tempo and same_notes(again.notes, view.notes)
