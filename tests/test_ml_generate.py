"""Phase C3: the token grammar used while writing, and the chart scores."""
from fractions import Fraction as F

import numpy as np

from backend.chartcodec.notes import Tap
from backend.chartcodec.parse import parse_chart
from backend.chartcodec.tokens import INDEX, decode, encode, from_ids, ids
from ml import score
from ml.grammar import Grammar


def test_random_writing_under_the_grammar_always_decodes():
    rng = np.random.default_rng(3)
    for _ in range(60):
        g, seq = Grammar(), [INDEX["BOS"]]
        while not g.done:
            m = g.allowed()
            if len(seq) > 300 and m[INDEX["EOS"]]:
                t = INDEX["EOS"]
            else:
                m2 = m.copy(); m2[INDEX["EOS"]] = False
                t = int(rng.choice(np.flatnonzero(m2 if m2.any() else m)))
            seq.append(t); g.push(t)
        decode(from_ids(seq))                               # raises if invalid


def test_grammar_accepts_real_charts_and_blocks_impossible_slides():
    c, _ = parse_chart("(150){8}1/5,2-6[8:1],3>5[8:1]*-7[8:1],Ch[4:1]/E2f,{20}4,5,6,7,8,{4}1V35[4:1],E")
    g = Grammar()
    for t in ids(encode(c.notes))[1:]:
        assert g.allowed()[t]
        g.push(t)
    g = Grammar()
    for name in ["BEAT", "P0", "SLIDE", "B1", "PATH", "SH_-"]:
        g.push(INDEX[name])
    m = g.allowed()
    assert not m[INDEX["B2"]] and not m[INDEX["B8"]] and not m[INDEX["B1"]] and m[INDEX["B5"]]   # 1-2 is impossible


def test_scores():
    real = [Tap(F(k, 2), (k % 8) + 1) for k in range(64)]
    assert score.timing_f1(np.arange(10.0), np.arange(10.0)) == 1.0
    assert score.timing_f1(np.arange(10.0), np.arange(10.0) + 0.5) == 0.0
    model = score.flow_model([real])                        # stairs only: +1 after +1
    stairs, rnd = score.flow_ll(real, model), score.flow_ll(score.randomized(real, 1), model)
    assert stairs > rnd
    assert score.repeat_share(real) > 0.5                   # the same stairs every bar
    toks = ids(encode(real))
    rot = ids(encode([Tap(n.beat, (n.button + 2) % 8 + 1) for n in real]))
    share, longest = score.copied(rot, score.shingles(toks))
    assert share == 1.0 and longest == len(toks)           # a rotated copy is caught


def test_loop_guard_triggers_only_after_three_repeats():
    from ml.generate import _guard_target
    bars = {0: [5, 6, 7], 1: [5, 6, 7], 2: [5, 6, 7]}
    assert _guard_target(bars, 3, []) == 5 and _guard_target(bars, 3, [5]) == 6
    assert _guard_target(bars, 3, [9]) is None                       # already different: leave it
    assert _guard_target({0: [5, 6], 1: [5, 6]}, 2, []) is None      # 2 repeats are fine
    assert _guard_target({0: [1], 1: [2], 2: [1], 3: [2], 4: [1], 5: [2]}, 6, []) == 1   # A B A B A B
