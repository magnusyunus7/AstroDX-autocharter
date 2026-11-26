"""Phase C2: dataset build (labels, splits, sync check, one small build)."""
import json
import shutil

import numpy as np
import pytest

from ml import charts, sync


def test_labels():
    assert charts.era_of("13. FiNALE") == "STD" and charts.era_of("14. DX") == "DX"
    assert charts.source_of("19. UNiVERSE PLUS", 5) == "converted"
    assert charts.source_of("20. FESTiVAL", 5) == "proofread"
    assert charts.source_of("20. FESTiVAL", 3) == "converted"          # Basic/Advanced: still dumped
    assert [charts.level_value(t) for t in ("13", "13+", "14.6", "?", "")] == [13.0, 13.7, 14.6, None, None]
    assert charts.title_key("RONDØ ") == charts.title_key("rondø")


def test_split_is_deterministic_and_about_90_5_5():
    from ml.build_dataset import split_of
    keys = [f"song{i}" for i in range(4000)]
    s = [split_of(k) for k in keys]
    assert s == [split_of(k) for k in keys]
    share = {x: s.count(x) / len(s) for x in ("train", "val", "test")}
    assert 0.87 < share["train"] < 0.93 and 0.03 < share["val"] < 0.07 and 0.03 < share["test"] < 0.07


def test_read_song_records():
    text = ("&title=T\n&first=1.0\n&lv_5=14+\n&inote_5=(120){8}1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,E\n"
            "&lv_4=12\n&inote_4=(120)E\n&lv_3=?\n&inote_3=(120){4}1,2,E\n&lv_2=5\n&inote_2=(120){4}9,E\n")
    song, recs = charts.read_song(text, "24. PRiSM")
    by = {r["slot"]: r for r in recs}
    assert 4 not in by                                                  # empty placeholder skipped
    assert by[5]["use"] and by[5]["level_value"] == 14.7 and by[5]["fit"] == 1.0 and by[5]["notes"] == 16
    assert not by[3]["use"] and "level" in by[3]["reason"]
    assert not by[2]["use"] and by[2]["reason"].startswith("parse")


def test_sync_finds_the_offset():
    fps = 344.5
    env = np.zeros(int(60 * fps))
    true = np.arange(1.0, 58.0, 0.25)
    for t in true + 0.040:                                              # audio hits 40 ms after the notes
        env[int(round(t * fps))] = 1.0
    r = sync.chart_offset(env, fps, true)
    assert abs(r["offset_ms"] - 40) <= 3 and r["sharpness"] > 4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_small_build(tmp_path):
    """Synthetic songs: three in sync, one whose chart is 60 ms early -> resynced."""
    from tools.make_test_song import make, write_mp3
    from ml import build_dataset
    coll = tmp_path / "coll"
    for name, first in (("InSync", 1.0), ("InSync2", 1.0), ("InSync3", 1.0), ("Early", 0.94)):
        d = coll / "24. PRiSM" / name
        d.mkdir(parents=True)
        write_mp3(make(150, 1.0, 24, seed=3), d / "track.mp3")
        body = "(150){4}" + "1,5," * 40 + "E"
        (d / "maidata.txt").write_text(f"&title={name}\n&first={first}\n&lv_5=13\n&inote_5={body}\n", encoding="utf-8")
    out = tmp_path / "ds"
    (out / "songs").mkdir(parents=True)
    import argparse
    args = argparse.Namespace(collection=coll, out=out, retime=None, workers=1, no_bpm_test=False, limit=None)
    build_dataset.build(args, out)
    rep = build_dataset.summarize(out)
    rows = {json.loads(l)["song"]: json.loads(l) for l in (out / "charts.jsonl").read_text().splitlines()}
    assert rows["InSync"]["use"] and abs(rows["InSync"]["shift_ms"]) < 1
    assert rows["Early"]["use"] and 45 <= rows["Early"]["shift_ms"] <= 75     # notes moved onto the audio
    assert rep["bpm_test"]["relation"] == {"exact": 4}
    assert rep["charts_used"] == 4                                      # same notes, different songs: all kept
    assert build_dataset.verify(out) == 0
