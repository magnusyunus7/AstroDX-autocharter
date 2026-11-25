"""Phase C1 check: run every chart in the collection through the chart codec.

    python -m tools.check_codec "G:/My Drive/<collection folder>"

For every maidata.txt (utage folders "[..." and the excluded version folders
are skipped, and so is slot 7) and every difficulty in it:
  1. parse the simai text exactly (known typos fixed and counted),
  2. write it back to simai and parse that again -> must give the same chart,
  3. take the model's view (odd values moved to the grid, counted), turn it
     into tokens and back -> must give the same notes,
  4. rotate and mirror it (all 16 as tokens, two also written to simai and
     read back) -> must still work.
Steps 2-4 failing means a bug in the codec. Step 1 failing means syntax the
parser doesn't know yet (or a typo it can't guess): listed with examples.

It also measures how well each chart's notes follow its own beat grid: the
share of note moments within 3 ms of a 1/48-beat or 5th/7th/9th/10th
position. Charts far below 100% (old converted charts that squeezed a tempo
change into odd divisors, like だんだん早くなる) are listed: their beats don't
follow the music, which C2 has to fix from the audio.

Only reads maidata.txt files (no audio), so it's quick even on a streamed
Google Drive folder. Writes a full report to data/codec_report.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fractions import Fraction                                          # noqa: E402

from backend.chartcodec.augment import transform                      # noqa: E402
from backend.chartcodec.notes import ChartData, same_notes, sort_key  # noqa: E402
from backend.chartcodec.parse import ParseError, parse_chart, parse_maidata  # noqa: E402
from backend.chartcodec.tokens import VOCAB, decode, encode             # noqa: E402
from backend.chartcodec.write import WriteError, write_chart            # noqa: E402

EXCLUDE = ("02.5. 舞萌中国", "25A. KALEIDXSCOPE Playlists", "legacy_maidata (DO NOT DOWNLOAD)")
SLOTS = range(1, 7)                       # 7 = utage
STD_UP_TO = 13                            # versions 01-13 (up to FiNALE) are STD, 14 (DX) on are DX
FIT_DENOMS = (48, 5, 7, 9, 10)
VARIANTS = ((3, False), (5, True))        # rotations written to simai and read back (all 16 are encoded)


def find_maidata(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root).parts
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE and not d.startswith("["))
        if "maidata.txt" in filenames and rel:
            yield Path(dirpath) / "maidata.txt"


def group_of(msg: str) -> str:
    """Error message without the quoted chart text, so equal problems group together."""
    return re.sub(r"'[^']*'|\"[^\"]*\"|\d+", "…", msg)


def era_of(song: str) -> str:
    m = re.match(r"(\d+)", song)
    return "STD" if m and int(m.group(1)) <= STD_UP_TO else "DX"


def grid_fit(chart) -> float:
    """Share of note moments within 3 ms of a 1/48-beat or 5th/7th/9th/10th position."""
    beats = sorted({n.beat for n in chart.notes})
    ok = 0
    for b in beats:
        ms = min(abs(float(b - Fraction(round(b * d), d))) for d in FIT_DENOMS) * 60000 / chart.bpm_at(b)
        ok += ms <= 3
    return ok / max(1, len(beats))


def check_chart(body: str, wholebpm):
    """-> (chart, issues, problem or None, model view, values moved). problem = (stage, message)."""
    try:
        chart, issues = parse_chart(body, wholebpm, repair=True)
    except ParseError as e:
        return None, None, ("parse", str(e)), None, 0
    except Exception as e:                                            # a crash is a codec bug
        return None, None, ("parse crash", f"{type(e).__name__}: {e}"), None, 0
    cv, moved_values = chart.model_view()
    view = cv.notes
    try:
        try:
            again, _ = parse_chart(write_chart(chart))
            if again.tempo != chart.tempo or not same_notes(again.notes, chart.notes):
                return chart, issues, ("write", "written text reads back differently"), None, 0
        except WriteError:                                            # too odd for simai text:
            issues.note("exact write impossible (model view written)")
        again, _ = parse_chart(write_chart(cv))                        # what generated charts look like
        if again.tempo != cv.tempo or not same_notes(again.notes, cv.notes):
            return chart, issues, ("write", "model view reads back differently"), None, 0
    except Exception as e:
        return chart, issues, ("write", f"{type(e).__name__}: {e}"), None, 0
    try:
        if decode(encode(view)) != view:
            return chart, issues, ("tokens", "tokens decode to different notes"), None, 0
    except Exception as e:
        return chart, issues, ("tokens", f"{type(e).__name__}: {e}"), None, 0
    try:
        key = lambda ns: sorted(ns, key=sort_key)
        for r, m in VARIANTS:
            moved = ChartData(cv.tempo, key(transform(view, r, m)))
            back, _ = parse_chart(write_chart(moved))
            if not same_notes(back.notes, moved.notes):
                return chart, issues, ("rotate/mirror", f"rotation {r}, mirror {m} doesn't read back"), None, 0
        for r in range(8):                                            # all 16 must encode
            for m in (False, True):
                moved = key(transform(view, r, m))
                if decode(encode(moved)) != moved:
                    return chart, issues, ("rotate/mirror", f"rotation {r}, mirror {m} tokens differ"), None, 0
        if key(transform(transform(chart.notes, 0, True), 0, True)) != chart.notes:
            return chart, issues, ("rotate/mirror", "mirroring twice isn't the identity"), None, 0
    except Exception as e:
        return chart, issues, ("rotate/mirror", f"{type(e).__name__}: {e}"), None, 0
    return chart, issues, None, view, moved_values


def check_song(args):
    """All charts of one maidata.txt -> list of per-chart results (runs in a worker)."""
    root, f = args
    song = str(f.parent.relative_to(root))
    try:
        meta = parse_maidata(f.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception as e:
        return song, None, f"{type(e).__name__}: {e}"
    out = []
    for slot, c in sorted(meta["charts"].items()):
        if slot not in SLOTS:
            out.append({"slot": slot, "skip": "slot"})
            continue
        if not c["body"].strip().strip("E").strip():
            out.append({"slot": slot, "skip": "empty"})
            continue
        chart, issues, problem, view, moved = check_chart(c["body"], meta["wholebpm"])
        r = {"slot": slot, "level": c["level"], "problem": problem}
        if not problem:
            r.update(notes=len(chart.notes), bpm_changes=len(chart.tempo) > 1, info=dict(issues.count),
                     moved=moved, fit=grid_fit(chart), tokens=encode(view))
        out.append(r)
    return song, out, None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "codec_report.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    root = args.folder.expanduser()
    if not root.is_dir():
        sys.exit(f"not a folder: {root}")

    print("finding maidata.txt files ...", flush=True)
    files = list(find_maidata(root))
    print(f"{len(files)} songs", flush=True)

    stats = Counter()
    problems = defaultdict(list)            # (stage, group) -> examples
    problem_n = Counter()                   # (stage, group) -> how many charts
    info = Counter()
    eras = Counter()
    off_grid = []                           # (fit, song, slot)
    lengths, used = [], Counter()
    levels = Counter()
    t0 = time.time()
    pool = mp.Pool(args.workers)
    try:
        for k, (song, results, err) in enumerate(pool.imap_unordered(check_song, [(root, f) for f in files])):
            if err:
                problems[("read", "unreadable maidata")].append({"song": song, "message": err})
                problem_n[("read", "unreadable maidata")] += 1
                stats["songs unreadable"] += 1
                continue
            stats["songs"] += 1
            for r in results:
                slot = r["slot"]
                if r.get("skip") == "slot":
                    stats["charts skipped (slot 7 / other)"] += 1
                    continue
                if r.get("skip") == "empty":
                    stats["charts empty"] += 1
                    continue
                stats["charts"] += 1
                if r["problem"]:
                    stage, msg = r["problem"]
                    stats[f"failed: {stage}"] += 1
                    ex = problems[(stage, group_of(msg))]
                    problem_n[(stage, group_of(msg))] += 1
                    if len(ex) < 8:
                        ex.append({"song": song, "slot": slot, "message": msg})
                    continue
                stats["charts ok"] += 1
                eras[era_of(song)] += 1
                levels[r["level"]] += 1
                stats["notes"] += r["notes"]
                if r["bpm_changes"]:
                    stats["charts with BPM changes"] += 1
                info.update(r["info"])
                if any(k.startswith("fixed") for k in r["info"]):
                    stats["charts with typos fixed"] += 1
                if r["moved"]:
                    stats["charts with odd values (moved for the model)"] += 1
                    stats["odd values moved for the model"] += r["moved"]
                if r["fit"] < 0.98:
                    off_grid.append((round(r["fit"], 3), song, slot))
                lengths.append(len(r["tokens"]))
                used.update(r["tokens"])
            if (k + 1) % 100 == 0:
                rate = (time.time() - t0) / (k + 1)
                print(f"  {k + 1}/{len(files)} songs, {stats['charts ok']} charts ok, "
                      f"{sum(v for s, v in stats.items() if s.startswith('failed'))} failed, "
                      f"about {rate * (len(files) - k - 1) / 60:.0f} min left", flush=True)
        pool.close()
    except KeyboardInterrupt:
        pool.terminate()
        print("\nstopped early: the summary below covers the songs read so far")

    lengths.sort()
    pct = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))] if lengths else 0
    report = {
        "folder": str(root), "stats": dict(stats), "info": dict(info), "eras": dict(eras),
        "beat_grid_off": [{"fit": f, "song": s, "slot": sl} for f, s, sl in sorted(off_grid)],
        "tokens_per_chart": {"median": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                             "max": lengths[-1] if lengths else 0},
        "unused_tokens": [t for t in VOCAB if t not in used and t not in ("PAD",)],
        "token_counts": dict(used.most_common()),
        "levels": dict(sorted(levels.items())),
        "problems": [{"stage": s, "kind": g, "charts": problem_n[(s, g)], "examples": ex}
                     for (s, g), ex in sorted(problems.items(), key=lambda x: -problem_n[x[0]])],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== summary")
    for key in ("songs", "charts", "charts ok", "notes", "charts with BPM changes", "charts with typos fixed",
                "charts with odd values (moved for the model)", "odd values moved for the model",
                "charts empty", "charts skipped (slot 7 / other)", "songs unreadable"):
        if stats.get(key):
            print(f"  {key}: {stats[key]}")
    for key, v in sorted(stats.items()):
        if key.startswith("failed"):
            print(f"  {key}: {v}")
    print(f"  eras: {dict(eras)}; also noted: {dict(info)}")
    broken = [x for x in off_grid if x[0] < 0.9]
    print(f"  beat grid off the music: {len(broken)} charts under 90% fit, "
          f"{len(off_grid) - len(broken)} at 90-98% (listed in the report)")
    print(f"  tokens per chart: median {pct(0.5)}, 90% {pct(0.9)}, 99% {pct(0.99)}, max {report['tokens_per_chart']['max']}")
    if problems:
        print("\n=== problems (up to 8 examples each)")
        for (s, g), ex in sorted(problems.items(), key=lambda x: -problem_n[x[0]]):
            print(f"  [{s}] {problem_n[(s, g)]} chart(s), e.g. {ex[0]['message']}")
            for e in ex[:3]:
                print(f"      {e['song']}  slot {e.get('slot', '-')}")
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
