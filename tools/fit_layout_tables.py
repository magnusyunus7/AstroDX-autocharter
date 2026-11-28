"""Learn from the real charts how charters scale density with the music, for
the layout generator. Runs once on the cluster, over the C2 dataset:

    python tools/fit_layout_tables.py --data ~/dataset --out ~/layout_tables.json

What it learns (from training charts; validation charts only for the report):

  density   per difficulty group and per kind of bar (5 loudness x 5 hit-
            strength bins, both relative to the rest of the same song): how
            many note moments the bar gets compared to the chart's average,
            how often it is left empty, and its share of holds / slides /
            EACH. Plus the chart's average moments per bar by BPM.
  slots     chance that a 16th-note slot gets a note, by its place in the bar
            (downbeat / beat / 8th / 16th), how strong the hit there is, and
            how busy the bar is meant to be.
  flow      how one hand move follows the previous one (stairs, trills,
            spins ...), as in ml/score.flow_model.
  model     a small linear model of bar density from more measures than the
            table (strong hits per bar, drums/bass, brightness, the
            surrounding section, position in the song); see
            backend/layout/features.bar_matrix. Used smoothed over 3 bars.

Difficulty groups: basic (below 7), light (7 to 9), medium (10 to 12),
dense (13 and harder).

The report (also printed) checks the density table on validation charts:
does the predicted bar density follow the real chart better than "a note on
every strong peak" (the old way)? r = correlation per chart, averaged;
"calm bars" = the song's quietest quarter of bars, density compared to the
chart's average (real charts drop a lot there). "human" = the same song's
other chart (nearest difficulty slot) as the prediction: how much two real
charts agree, the most a prediction can be expected to reach.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from fractions import Fraction
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chartcodec.notes import Hold, Slide, TouchHold         # noqa: E402
from backend.chartcodec.tokens import decode, from_ids              # noqa: E402
from backend.layout import features as F                            # noqa: E402
from ml.data import beats_to_seconds, load_rows                     # noqa: E402
from ml.score import flow_steps                                     # noqa: E402

GROUPS = ("basic", "light", "medium", "dense")
GROUP_FROM = (0.0, 7.0, 10.0, 13.0)
DENS_EDGES = (0.5, 1.0, 1.5)          # bar density (vs chart average) -> 4 bins for the slot table
BPM_EDGES = (120.0, 160.0, 200.0)     # -> 4 BPM bins for the average density
PEAK = 0.75                           # "every peak" baseline: slots stronger than 75% of the song's slots
MIN_CELL = 30                         # fewer bars than this in a cell -> fall back to the loudness-only row
RIDGE = 10.0                          # model fit: weight shrinkage
MAX_RATIO = 3.0                       # model fit: bars busier than 3x average count as 3x


def group_of(level: float) -> int:
    return max(i for i, g in enumerate(GROUP_FROM) if level >= g)


def bpm_bin(bpm: float) -> int:
    return int(np.searchsorted(BPM_EDGES, bpm, side="right"))


def one_chart(row, tok_all, cv):
    """Bars and slots of one chart. Returns None for charts with too few bars."""
    tok = [int(x) for x in tok_all[row["token_start"]:row["token_start"] + row["token_count"]]]
    notes = decode(from_ids(tok))
    if len(notes) < 16:
        return None
    tempo = [(float(Fraction(b)), v) for b, v in row["tempo"]]
    shift = row.get("shift_ms", 0.0) / 1000.0
    first = row["_first"]

    by_beat = defaultdict(list)
    for n in notes:
        by_beat[n.beat].append(n)
    beats = sorted(by_beat)
    b0, b1 = int(beats[0] // 4), int(beats[-1] // 4)
    nb = b1 - b0 + 1
    if nb < 8:
        return None

    moments = np.zeros(nb); holds = np.zeros(nb); slides = np.zeros(nb); each = np.zeros(nb)
    n_notes = np.zeros(nb)
    on_grid = np.zeros(nb * 16, bool)
    offgrid = 0
    for b in beats:
        k = int(b // 4) - b0
        ns = by_beat[b]
        moments[k] += 1
        n_notes[k] += len(ns)
        each[k] += len(ns) > 1
        holds[k] += sum(isinstance(n, (Hold, TouchHold)) for n in ns)
        slides[k] += sum(isinstance(n, Slide) for n in ns)
        q = Fraction(b) * 4
        if q.denominator == 1:
            on_grid[int(q) - b0 * 16] = True
        else:
            offgrid += 1

    bar_beats = 4.0 * np.arange(b0, b1 + 2)
    bt = first + shift + beats_to_seconds(tempo, bar_beats)
    rate = F.bar_ratings(cv, bt[:-1], bt[1:])

    slot_beats = b0 * 4 + np.arange(nb * 16) / 4.0
    st = first + shift + beats_to_seconds(tempo, slot_beats)
    strength = F.percentile(F.slot_strength(cv, st))
    peaks = (strength >= PEAK).reshape(nb, 16).sum(1)

    X = F.bar_matrix(cv, bt[:-1], bt[1:], peaks)
    mean = moments.mean()
    return {
        "X": X.astype(np.float32), "b0": b0,
        "ratio": moments / mean, "moments": moments, "empty": moments == 0,
        "hold": holds / np.maximum(1, n_notes), "slide": slides / np.maximum(1, n_notes),
        "each": each / np.maximum(1, moments),
        "loud": rate["loud"], "loud_bin": rate["loud_bin"], "hit_bin": rate["hit_bin"],
        "peaks": peaks.astype(np.float64),
        "slot_metric": F.metric_class(np.arange(nb * 16) % 16), "slot_str": F.to_bin(strength),
        "slot_note": on_grid, "slot_dens": np.repeat(np.searchsorted(DENS_EDGES, moments / mean, side="right"), 16),
        "mean_per_bar": float(mean), "bpm": tempo[0][1], "offgrid": offgrid, "n_moments": len(beats),
        "flow": flow_steps(notes),
    }


def one_song(job):
    sid, rows, data = job
    try:
        with np.load(Path(data) / "songs" / f"{sid}.npz") as z:
            mel, tok_all, first = z["mel"], z["tokens"], float(z["first"])
    except Exception as e:                                           # noqa: BLE001
        return sid, None, f"{type(e).__name__}: {e}"
    cv = F.curves(mel)
    out = []
    for r in rows:
        r = dict(r, _first=first)
        try:
            c = one_chart(r, tok_all, cv)
        except Exception as e:                                       # noqa: BLE001
            return sid, None, f"{r['slot']}: {type(e).__name__}: {e}"
        if c is not None:
            c.update(split=r["split"], group=group_of(r["level_value"]), level=r["level_value"], slot=r["slot"],
                     song_key=sid, name=f"{r['version']}/{r['song']} slot {r['slot']}")
            out.append(c)
    return sid, out, None


# ------------------------------------------------------------------ fitting

def fit(charts):
    G, B = len(GROUPS), F.BINS
    cells = np.zeros((G, B, B, 6)); cell_n = np.zeros((G, B, B))      # ratio, empty, hold, slide, each, ratio^2
    rows_ = np.zeros((G, B, 6)); row_n = np.zeros((G, B))
    slot_tot = np.zeros((G, 4, 4, B)); slot_hit = np.zeros((G, 4, 4, B))
    flow = np.ones((G, 8, 8))
    XtX = np.zeros((G, F.N_INPUTS, F.N_INPUTS)); Xty = np.zeros((G, F.N_INPUTS))
    per_bar = defaultdict(list)
    off = np.zeros((G, 2))
    for c in charts:
        g = c["group"]
        v = np.stack([c["ratio"], c["empty"], c["hold"], c["slide"], c["each"], c["ratio"] ** 2], 1)
        np.add.at(cells[g], (c["loud_bin"], c["hit_bin"]), v)
        np.add.at(cell_n[g], (c["loud_bin"], c["hit_bin"]), 1)
        np.add.at(rows_[g], c["loud_bin"], v)
        np.add.at(row_n[g], c["loud_bin"], 1)
        idx = (c["slot_dens"], c["slot_metric"], c["slot_str"])
        np.add.at(slot_tot[g], idx, 1)
        np.add.at(slot_hit[g], idx, c["slot_note"].astype(np.float64))
        st = c["flow"]
        for x, y in zip(st, st[1:]):
            flow[g, x + 3, y + 3] += 1
        per_bar[(g, bpm_bin(c["bpm"]))].append(c["mean_per_bar"])
        off[g] += (c["offgrid"], c["n_moments"])
        X = c["X"].astype(np.float64)
        XtX[g] += X.T @ X
        Xty[g] += X.T @ np.minimum(c["ratio"], MAX_RATIO)

    tables = {"version": 1, "groups": dict(zip(GROUPS, GROUP_FROM)), "bins": B,
              "dens_edges": list(DENS_EDGES), "bpm_edges": list(BPM_EDGES), "metric": list(F.METRIC),
              "density": {}, "slots": {}, "flow": {}, "model": {}, "model_smooth": 1}
    for g, name in enumerate(GROUPS):
        mean_cell = cells[g] / np.maximum(1, cell_n[g])[..., None]
        mean_row = rows_[g] / np.maximum(1, row_n[g])[..., None]
        use = cell_n[g] >= MIN_CELL
        table = np.where(use[..., None], mean_cell, mean_row[:, None, :])
        keys = ("ratio", "empty", "hold", "slide", "each")
        tables["density"][name] = {
            **{k: np.round(table[..., i], 4).tolist() for i, k in enumerate(keys)},
            "ratio_sd": np.round(np.sqrt(np.maximum(0, table[..., 5] - table[..., 0] ** 2)), 4).tolist(),
            "n": cell_n[g].astype(int).tolist(),
            "per_bar_by_bpm": [round(float(np.median(per_bar[(g, k)])), 3) if per_bar[(g, k)] else None
                               for k in range(len(BPM_EDGES) + 1)],
            "charts_by_bpm": [len(per_bar[(g, k)]) for k in range(len(BPM_EDGES) + 1)],
            "offgrid_share": round(float(off[g, 0] / max(1, off[g, 1])), 4),
        }
        p = (slot_hit[g] + 0.5) / (slot_tot[g] + 1.0)
        tables["slots"][name] = {"p": np.round(p, 4).tolist(), "n": slot_tot[g].astype(int).tolist()}
        tables["flow"][name] = np.round(np.log(flow[g] / flow[g].sum(1, keepdims=True)), 4).tolist()
        reg = RIDGE * np.eye(F.N_INPUTS); reg[0, 0] = 0.0
        tables["model"][name] = np.round(np.linalg.solve(XtX[g] + reg + 1e-9 * np.eye(F.N_INPUTS), Xty[g]), 5).tolist()
    return tables


def _r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def _other(c, charts_by_song):
    """The same song's other chart nearest in difficulty slot, on the bars both cover."""
    others = [o for o in charts_by_song.get(c["song_key"], []) if o is not c and o["slot"] != c["slot"]]
    if not others:
        return None
    o = min(others, key=lambda o: (abs(o["slot"] - c["slot"]), o["slot"]))
    lo, hi = max(c["b0"], o["b0"]), min(c["b0"] + len(c["ratio"]), o["b0"] + len(o["ratio"]))
    if hi - lo < 8:
        return None
    full = np.full(len(c["ratio"]), np.nan)
    full[lo - c["b0"]:hi - c["b0"]] = o["ratio"][lo - o["b0"]:hi - o["b0"]]
    return full


def report(tables, charts):
    by_song = defaultdict(list)
    for c in charts:
        by_song[c["song_key"]].append(c)
    out = {}
    for g, name in enumerate(GROUPS):
        ratio = np.array(tables["density"][name]["ratio"])
        w = np.array(tables["model"][name])
        ways = ("table", "model", "every_peak", "human")
        r1, r4 = defaultdict(list), defaultdict(list)
        calm = defaultdict(list)
        for c in (c for c in charts if c["group"] == g):
            real = c["ratio"]
            preds = {"table": ratio[c["loud_bin"], c["hit_bin"]],
                     "model": F.predict(w, c["X"].astype(np.float64), tables["model_smooth"]),
                     "every_peak": c["peaks"], "human": _other(c, by_song)}
            q = c["loud"] < 0.25
            for k, p in preds.items():
                if p is None:
                    continue
                ok = ~np.isnan(p)
                a = _r(real[ok], p[ok])
                if a is not None:
                    r1[k].append(a)
                n4 = ok.sum() // 4 * 4
                if n4 >= 16:
                    s4 = lambda x: x[ok][:n4].reshape(-1, 4).sum(1)                 # noqa: E731
                    a = _r(s4(real), s4(p))
                    if a is not None:
                        r4[k].append(a)
                if k != "human" and (q & ok).sum() >= 2 and p[ok].mean() > 0:
                    calm[k].append(p[q & ok].mean() / p[ok].mean())
            if q.sum() >= 2:
                calm["real"].append(real[q].mean())
        out[name] = {
            "charts": len(r1["table"]),
            "r_bar": {k: _avg(r1[k]) for k in ways},
            "r_4bars": {k: _avg(r4[k]) for k in ways},
            "calm_bars_density": {k: _avg(calm[k]) for k in ("real", "table", "model", "every_peak")},
        }
    return out


def _avg(x):
    x = list(x)
    return round(float(np.mean(x)), 3) if len(x) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="C2 dataset folder (charts.jsonl, songs/)")
    ap.add_argument("--out", type=Path, required=True, help="layout_tables.json to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N songs (quick trial)")
    args = ap.parse_args()

    rows = load_rows(args.data, "train") + load_rows(args.data, "val")
    songs = defaultdict(list)
    for r in rows:
        songs[r["id"]].append(r)
    jobs = [(sid, rs, str(args.data)) for sid, rs in sorted(songs.items())]
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} songs, {sum(len(j[1]) for j in jobs)} charts (train + val)", flush=True)

    charts, failed = [], []
    with Pool(args.workers) as pool:
        for k, (sid, out, err) in enumerate(pool.imap_unordered(one_song, jobs, chunksize=4)):
            if err:
                failed.append(f"{sid}: {err}")
            else:
                charts += out
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(jobs)} songs", flush=True)

    train = [c for c in charts if c["split"] == "train"]
    val = [c for c in charts if c["split"] == "val"]
    print(f"fitting on {len(train)} training charts, checking on {len(val)} validation charts", flush=True)
    tables = fit(train)
    tables["report"] = report(tables, val)
    tables["report"]["failed_songs"] = failed[:50]
    args.out.write_text(json.dumps(tables, ensure_ascii=False), encoding="utf-8")

    print("\n correlation with the real chart's bar density (per bar | per 4 bars)")
    print(" group    charts    table        model        every peak   human")
    for name in GROUPS:
        x = tables["report"][name]
        cells = "  ".join(f"{x['r_bar'][k]!s:>5}|{x['r_4bars'][k]!s:<5}" for k in ("table", "model", "every_peak", "human"))
        print(f" {name:<8} {x['charts']:>5}    {cells}")
    print("\n calm bars (song's quietest quarter), density vs the chart's average")
    print(" group    real   table  model  every peak")
    for name in GROUPS:
        c = tables["report"][name]["calm_bars_density"]
        print(f" {name:<8} {c['real']!s:>5}  {c['table']!s:>5}  {c['model']!s:>5}  {c['every_peak']!s:>5}")
    print("\n average moments per bar by BPM (<120, 120-160, 160-200, 200+):")
    for name in GROUPS:
        d = tables["density"][name]
        print(f" {name:<8} {d['per_bar_by_bpm']}   off the 16th grid: {100 * d['offgrid_share']:.1f}% of moments")
    if failed:
        print(f"\n{len(failed)} songs failed, first: {failed[0]}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
