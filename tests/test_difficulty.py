import pytest

from backend.generator.difficulty import LEVELS, level_value, targets


def test_levels_parse_and_order():
    assert level_value("14") == pytest.approx(14.2)
    assert level_value("14+") == pytest.approx(14.7)
    with pytest.raises(ValueError):
        level_value("fourteen")
    vals = [level_value(l) for l in LEVELS]
    assert vals == sorted(vals)


def test_targets_are_monotonic_and_match_real_14s():
    ts = [targets(l) for l in LEVELS]
    for a, b in zip(ts, ts[1:]):
        assert b.nps > a.nps and b.peak_nps > a.peak_nps
        assert b.longest_stream >= a.longest_stream
        assert b.each_pct <= a.each_pct
    t14, t14p = targets("14"), targets("14+")
    assert 6.0 < t14.nps < 7.0 and 7.0 < t14p.nps < 8.0          # your 14 / 14+ charts
    assert 13 < t14.peak_nps < 15.5 and 15 < t14p.peak_nps < 17
    assert targets("11").longest_stream == 0                      # no streams below ~12