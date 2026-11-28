"""Layout generator: audio measurements and the table fitting (tools/fit_layout_tables.py)."""
from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chartcodec.notes import Tap                                # noqa: E402
from backend.chartcodec.tokens import encode, ids                       # noqa: E402
from backend.layout import features as F                                # noqa: E402

BPM = 120.0                      # 2 beats per second, a bar = 2 s


def fake_mel(seconds: float, calm: tuple[float, float], seed: int = 0) -> np.ndarray:
    """Hits on every 8th note; the calm stretch is 25 dB quieter but its hits
    are still sharp peaks (like a soft piano part)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * F.FPS)
    m = np.full((n, 80), -60.0) + rng.normal(0, 0.5, (n, 80))
    t = np.arange(n) / F.FPS
    quiet = (t >= calm[0]) & (t < calm[1])
    m[~quiet] += 30.0
    for h in np.arange(0, seconds, 0.25):
        f = int(round(h * F.FPS))
        if f < n:
            m[f:f + 3] += 20.0
    return np.clip(m, -80, 0).astype(np.float16)


def test_percentile_ranks_and_ties():
    p = F.percentile(np.array([3.0, 1.0, 2.0, 2.0]))
    assert p[1] == 0.0 and p[0] == 1.0 and p[2] == p[3]
    assert np.all(F.to_bin(np.array([0.0, 0.19, 0.2, 0.99, 1.0])) == [0, 0, 1, 4, 4])


def test_calm_bars_rank_low_even_with_sharp_peaks():
    mel = fake_mel(40.0, calm=(16.0, 24.0))
    cv = F.curves(mel)
    t0 = np.arange(0, 40, 2.0)
    r = F.bar_ratings(cv, t0, t0 + 2.0)
    calm = (t0 >= 16) & (t0 < 24)
    assert r["loud"][calm].max() < r["loud"][~calm].min()
    assert np.all(r["loud_bin"][calm] == 0)


def test_slot_strength_finds_hits_and_metric_classes():
    mel = fake_mel(8.0, calm=(0.0, 0.0))
    cv = F.curves(mel)
    on = F.slot_strength(cv, np.arange(0.25, 7.5, 0.25))
    off = F.slot_strength(cv, np.arange(0.25, 7.5, 0.25) + 0.12)
    assert on.min() > off.max()
    assert F.metric_class(np.arange(16)).tolist() == [0, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3]


def _chart(seconds, calm):
    """Real-chart-like: 8th notes in loud bars, one note per bar in calm bars."""
    notes, beat = [], Fraction(0)
    end = Fraction(int(seconds * BPM / 60))
    k = 0
    while beat < end:
        t = float(beat) * 60 / BPM
        quiet = calm[0] <= t < calm[1]
        if not quiet or beat % 4 == 0:
            notes.append(Tap(beat, 1 + k % 8)); k += 1
        beat += Fraction(4) if quiet else Fraction(1, 2)
    return notes


def test_fit_tables_end_to_end(tmp_path: Path):
    data = tmp_path / "ds"
    (data / "songs").mkdir(parents=True)
    rows = []
    for i in range(6):
        sid = f"s{i}"
        calm = (16.0 + 2 * i, 28.0 + 2 * i)
        mel = fake_mel(60.0, calm, seed=i)
        tok = np.array(ids(encode(_chart(58.0, calm))), np.int16)
        np.savez(data / "songs" / f"{sid}.npz", mel=mel, tokens=tok, first=np.float64(0.0), meta="{}")
        for split in ("train", "val") if i < 5 else ("val",):
            if split == "val" and i < 4:
                continue
            rows.append({"id": sid, "version": "x", "song": sid, "slot": 5, "split": split, "use": True,
                         "level_value": 13.5, "tempo": [["0", BPM]], "shift_ms": 0.0,
                         "token_start": 0, "token_count": int(len(tok))})
    (data / "charts.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "tables.json"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "fit_layout_tables.py"), "--data", str(data),
                        "--out", str(out), "--workers", "2"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = json.loads(out.read_text())
    d = t["density"]["dense"]
    ratio = np.array(d["ratio"])
    n = np.array(d["n"])
    assert ratio[0][n[0] > 0].mean() < 0.3                       # calm bars: far fewer notes
    assert np.array(t["slots"]["dense"]["p"]).shape == (4, 4, 5)
    assert np.array(t["flow"]["dense"]).shape == (8, 8)
    rep = t["report"]["dense"]
    assert rep["charts"] >= 1
    calm = rep["calm_bars_density"]
    assert calm["table"] < calm["every_peak"]                     # the table drops in calm parts, peaks don't
