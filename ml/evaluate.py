"""Phase C3: judge a trained model by the charts it writes.

    python -m ml.evaluate --data ~/dataset --collection ~/collection --model ~/runs/mid_reg/best.pt --out ~/eval/mid_reg

Picks validation songs that have a real chart at level 13 and up, writes a
chart for each (same song, same difficulty slot and level as the real one),
and scores it against the real chart (see ml/score.py). The real chart and a
random version of it (same rhythm, random buttons) are scored the same way,
so every number has a reference.

Writes:
  <out>/report.json        every chart's scores + the averages
  <out>/summary.txt        the averages as a small table
  <out>/charts/<song>/     maidata.txt (generated chart as Master; the real
                           one as Re:Master for side-by-side play) + track
Safe to stop and run again: finished charts are skipped.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch                                                             # noqa: E402

from backend.chartcodec.notes import ChartData                          # noqa: E402
from backend.chartcodec.tokens import decode, from_ids                  # noqa: E402
from ml import score                                                     # noqa: E402
from ml.data import SongStore, load_rows                                 # noqa: E402
from ml.generate import chart_text, generate, load_model, maidata       # noqa: E402


def pick(rows, n: int, min_level: float) -> list[dict]:
    """Up to n charts at min_level+, at most one per song (the hardest), spread
    over all versions (a fixed shuffle, so every model gets the same songs)."""
    import zlib
    best = {}
    for r in rows:
        if r["level_value"] >= min_level and r["slot"] in (4, 5, 6) and len(r["tempo"]) == 1:
            if r["id"] not in best or r["level_value"] > best[r["id"]]["level_value"]:
                best[r["id"]] = r
    out = sorted(best.values(), key=lambda r: zlib.crc32(r["id"].encode()))[:n]
    return sorted(out, key=lambda r: (r["version"], r["song"]))


def train_reference(data: Path, log=print) -> tuple[set[int], np.ndarray]:
    """From the training charts: the copy-check index and the flow model
    (charts at 12 and up). Built once, then loaded."""
    f, fm = data / "train_shingles.npy", data / "train_flow.npy"
    if f.exists() and fm.exists():
        return set(np.load(f).tolist()), np.load(fm)
    store = SongStore(data, cache=4)
    s, hard = set(), []
    rows = load_rows(data, "train")
    for k, r in enumerate(rows):
        _, _, tok, _ = store.get(r["id"])
        t = tok[r["token_start"]:r["token_start"] + r["token_count"]].tolist()
        s |= score.shingles(t)
        if r["level_value"] >= 12:
            hard.append(decode(from_ids(t)))
        if (k + 1) % 1000 == 0:
            log(f"  reference: indexed {k + 1}/{len(rows)} training charts", flush=True)
    model = score.flow_model(hard)
    np.save(f, np.fromiter(s, dtype=np.int64))
    np.save(fm, model)
    return s, model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--min-level", type=float, default=13.0)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--max-tokens", type=int, default=9000, help="(for quick tests)")
    ap.add_argument("--loop-guard", type=float, default=0.0, help="discourage endless repeats (try 2.0)")
    args = ap.parse_args()
    data, out, coll = args.data.expanduser(), args.out.expanduser(), args.collection.expanduser()
    (out / "charts").mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.model.expanduser(), device)
    print(f"model {ck['config']} from step {ck['step']}; device {device}", flush=True)
    rows = pick(load_rows(data, args.split), args.n, args.min_level)
    print(f"{len(rows)} {args.split} charts at level {args.min_level}+", flush=True)
    print("building the copy check and flow model (once) ...", flush=True)
    train_sh, flow = train_reference(data)
    store = SongStore(data, cache=4)
    index = {json.loads(l)["id"]: json.loads(l) for l in (data / "index.jsonl").read_text(encoding="utf-8").splitlines()}

    rep_f = out / "report.json"
    results = json.loads(rep_f.read_text()) if rep_f.exists() else {}
    for k, r in enumerate(rows):
        key = f"{r['id']}_{r['slot']}"
        if key in results:
            continue
        t0 = time.time()
        mel, mert, all_tok, first = store.get(r["id"])
        first += r.get("shift_ms", 0.0) / 1000.0
        tempo = [(float(Fraction(b)), v) for b, v in r["tempo"]]
        song = index[r["id"]]
        notes, toks, st = generate(model, device, mel, mert, first, tempo, song["duration"], r["level_value"],
                                   r["slot"], r["era"], r["source"], args.temperature, args.top_p, seed=k,
                                   max_tokens=args.max_tokens, loop_guard=args.loop_guard)
        real_tok = all_tok[r["token_start"]:r["token_start"] + r["token_count"]].tolist()
        real = decode(from_ids(real_tok))
        chart = ChartData([(Fraction(b), v) for b, v in r["tempo"]], real)
        n_bars = int(max([n.beat for n in real] + [0]) // 4) + 2
        rnd = score.randomized(real, seed=k)
        res = {"song": f"{r['version']}/{r['song']}", "slot": r["slot"], "level": r["level"], **st,
               "seconds": round(time.time() - t0, 1),
               "notes": {"gen": len(notes), "real": len(real)},
               "timing_f1": score.timing_f1(score.note_times(notes, chart, first), score.note_times(real, chart, first)),
               "density_r": score.density_r(notes, real, n_bars),
               "kinds": {"gen": score.kinds(notes), "real": score.kinds(real)},
               "flow": {"gen": score.flow_ll(notes, flow), "real": score.flow_ll(real, flow),
                        "random": score.flow_ll(rnd, flow)},
               "repeat": {"gen": score.repeat_share(notes), "real": score.repeat_share(real)},
               "problems": {"gen": score.problems_per_100(notes, chart, first),
                            "real": score.problems_per_100(real, chart, first),
                            "random": score.problems_per_100(rnd, chart, first)}}
        c, longest = score.copied(toks, train_sh)
        res["copied"] = {"share": c, "longest_tokens": longest}
        results[key] = res
        # playable files: generated as Master, the real chart as Re:Master
        d = out / "charts" / f"{r['version'][:2]} {r['song']}".replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        bpm = tempo[0][1]
        f_out, g_out, r_out = first, notes, real
        if first < 0:                                  # players want &first >= 0: start whole beats later
            from dataclasses import replace
            k = int(np.ceil(-first / (60.0 / bpm)))
            f_out = first + k * 60.0 / bpm
            g_out = [replace(n, beat=n.beat - k) for n in notes if n.beat >= k]
            r_out = [replace(n, beat=n.beat - k) for n in real if n.beat >= k]
        (d / "maidata.txt").write_text(maidata(
            f"{song['title']} [AI]", "chart model", f_out, bpm,
            {5: (r["level"], chart_text(g_out, tempo)), 6: (r["level"], chart_text(r_out, tempo))}), encoding="utf-8")
        src = coll / r["version"] / r["song"] / song["audio"]
        if src.exists() and not (d / f"track{src.suffix}").exists():
            shutil.copy(src, d / f"track{src.suffix}")
        rep_f.write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print(f"  {k + 1}/{len(rows)} {res['song']} ({r['level']}): {len(notes)} notes (real {len(real)}), "
              f"timing F1 {res['timing_f1']:.2f}, density r {res['density_r']:.2f}, "
              f"flow {res['flow']['gen'] or 0:.2f} (real {res['flow']['real'] or 0:.2f}, random {res['flow']['random'] or 0:.2f}), "
              f"problems/100 {res['problems']['gen']} (real {res['problems']['real']}), "
              f"copied {c:.0%} ({res['seconds']} s)", flush=True)
    summarize(results, out)


def summarize(results: dict, out: Path) -> None:
    R = list(results.values())
    avg = lambda f: float(np.mean([v for v in (f(r) for r in R) if v is not None])) if R else 0.0
    lines = [f"{len(R)} charts", "",
             f"{'':28}{'generated':>11}{'real':>9}{'random':>9}",
             f"{'notes (avg)':28}{avg(lambda r: r['notes']['gen']):11.0f}{avg(lambda r: r['notes']['real']):9.0f}",
             f"{'timing F1 vs real':28}{avg(lambda r: r['timing_f1']):11.2f}{1:9.2f}",
             f"{'density follows song (r)':28}{avg(lambda r: r['density_r']):11.2f}{1:9.2f}",
             f"{'hand flow like real (higher)':28}{avg(lambda r: r['flow']['gen']):11.2f}"
             f"{avg(lambda r: r['flow']['real']):9.2f}{avg(lambda r: r['flow']['random']):9.2f}",
             f"{'bars repeating a pattern':28}{avg(lambda r: r['repeat']['gen']):11.2f}{avg(lambda r: r['repeat']['real']):9.2f}",
             f"{'checker problems / 100':28}{avg(lambda r: r['problems']['gen']):11.1f}"
             f"{avg(lambda r: r['problems']['real']):9.1f}{avg(lambda r: r['problems']['random']):9.1f}",
             f"{'copied from training':28}{avg(lambda r: r['copied']['share']):11.0%}",
             f"{'longest copied (tokens)':28}{max([r['copied']['longest_tokens'] for r in R] or [0]):11}"
             f"  ({max(R, key=lambda r: r['copied']['longest_tokens'])['song'] if R else ''})",
             ""]
    for k in ("tap", "hold", "slide", "touch", "each", "break"):
        lines.append(f"{k + ' share':28}{avg(lambda r: r['kinds']['gen'][k]):11.1%}{avg(lambda r: r['kinds']['real'][k]):9.1%}")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    main()
