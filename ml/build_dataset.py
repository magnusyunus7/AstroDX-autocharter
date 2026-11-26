"""Phase C2: build the chart-model dataset from the collection.

    python -m ml.build_dataset --collection ~/collection --out ~/dataset

For every song (utage folders and the excluded versions are skipped):
  * every chart (slots 1-6) through the C1 codec, with labels (level, era,
    source) and checks (beat-grid fit, sync with the audio, audio long enough);
  * the audio decoded like the app does, and its log-mel spectrogram;
  * the app's BPM / offset detector run on the song and compared with the
    chart (how often generation would start from the right timing);
  * the song's split: train / val / test (90 / 5 / 5 by title, every copy of
    a song in the same split).
Writes <out>/songs/<id>.npz per song and <out>/index.jsonl, then a summary:
<out>/charts.jsonl (every chart, with use true/false and the reason, and
shift_ms: how far its notes are moved to line up with the audio, see below),
<out>/report.json, and a few pictures in <out>/pictures/ (notes drawn over
the spectrogram) to check by eye.

Sync is decided per song (all its charts share &first and the mp3): the
song's offset is the median of its charts with a clear answer. If those
charts agree (within SYNC_AGREE_MS) and the offset is more than SYNC_LIMIT_MS
from the collection's usual one (wrong &first for this mp3, a different
edit ...), the charts aren't dropped: they get shift_ms and training moves
their notes onto the audio. Listed in the report as "resynced". Songs whose
charts disagree are left as they are and listed as "sync unclear".

Safe to stop and run again: finished songs are skipped. Refuses to start if
another build is running on the same folder. Run with --summary-only to redo
the summary without touching the songs.

--retime <folder>: hand-retimed charts (data/retime layout:
<group>/<version> - <song>/maidata.txt) replace the collection's maidata.txt.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.chartcodec.tokens import decode, from_ids                  # noqa: E402
from ml import audio, charts, sync                                      # noqa: E402

EXCLUDE = ("02.5. 舞萌中国", "25A. KALEIDXSCOPE Playlists", "legacy_maidata (DO NOT DOWNLOAD)")
VAL, TEST = 5, 5                       # percent of song titles
SYNC_LIMIT_MS = 30.0                   # further than this from the usual offset: resynced
SYNC_SHARP = 4.0                       # only judge sync when the answer is this clear
SYNC_AGREE_MS = 15.0                   # a song's clear charts must agree this closely to resync it
AUDIO_SLACK = 1.0                      # the last note may be this far past the audio's end (s)
FORMAT = 1                             # bump when the song files change


def song_id(rel: str) -> str:
    return f"{zlib.crc32(rel.encode('utf-8')):08x}"


def split_of(key: str) -> str:
    h = zlib.crc32(("split:" + key).encode("utf-8")) % 100
    return "val" if h < VAL else "test" if h < VAL + TEST else "train"


def find_songs(collection: Path) -> list[Path]:
    out = []
    for f in sorted(collection.rglob("maidata.txt")):
        parts = f.relative_to(collection).parts[:-1]
        if any(p in EXCLUDE or p.startswith("[") for p in parts) or len(parts) < 2:
            continue
        out.append(f)
    return out


def retimed(retime_dir: Path | None, version: str, name: str) -> Path | None:
    if not retime_dir:
        return None
    for group in sorted(p for p in retime_dir.iterdir() if p.is_dir()):
        f = group / f"{version} - {name}" / "maidata.txt"
        if f.exists():
            return f
    return None


# ---------------------------------------------------------------- one song

def bpm_test(y, song: dict, recs: list[dict]) -> dict | None:
    """Run the app's detector and compare with the chart (highest difficulty
    without BPM changes)."""
    from backend.audio import tempo
    ref = [r for r in recs if "chart" in r and not r["bpm_changes"]]
    if not ref:
        return None
    r = max(ref, key=lambda r: r["slot"])
    bpm = r["chart"].tempo[0][1]
    t0 = time.time()
    det = tempo.detect(y, audio.SR)
    ratio = det.bpm / bpm
    rel = ("exact" if abs(ratio - 1) < 1e-3 else "x2" if abs(ratio - 2) < 2e-3 else
           "/2" if abs(ratio - 0.5) < 1e-3 else "x1.5" if abs(ratio - 1.5) < 2e-3 else
           "/1.5" if abs(ratio - 2 / 3) < 1e-3 else "wrong")
    beat = 60.0 / bpm
    d = (det.beat_phase - song["first"]) / beat
    phase_ms = (d - round(d)) * beat * 1000 if rel == "exact" else None
    return {"chart_bpm": bpm, "detected_bpm": det.bpm, "relation": rel,
            "phase_ms": None if phase_ms is None else round(phase_ms, 1), "seconds": round(time.time() - t0, 1)}


def build_song(job) -> dict:
    """One song -> its index record (and songs/<id>.npz written)."""
    maidata, collection, out, retime_dir, do_bpm = job
    rel_dir = maidata.parent.relative_to(collection)
    version, name = rel_dir.parts[0], "/".join(rel_dir.parts[1:])
    rel = rel_dir.as_posix()
    sid = song_id(rel)
    rec = {"id": sid, "version": version, "song": name, "format": FORMAT}
    try:
        src = retimed(retime_dir, version, name) or maidata
        text = src.read_text(encoding="utf-8-sig", errors="replace")
        song, recs = charts.read_song(text, version)
        rec.update(title=song["title"] or name, retimed=src != maidata)
        rec["title_key"] = charts.title_key(rec["title"])
        rec["split"] = split_of(rec["title_key"])
        tracks = sorted(maidata.parent.glob("track.*"))
        if not tracks:
            raise RuntimeError("no track.* audio")
        y = audio.decode(tracks[0])
        rec.update(audio=tracks[0].name, duration=round(len(y) / audio.SR, 3))
        m = audio.mel(y)
        env, fps = sync.onset_envelope(y, audio.SR)
        for r in recs:
            if "chart" not in r:
                continue
            times = np.array(sorted({song["first"] + r["chart"].seconds(n.beat) for n in r["chart"].notes}))
            r["last_note"] = round(float(times[-1]), 3)
            r.update(sync.chart_offset(env, fps, times, beat=60.0 / r["chart"].tempo[0][1]))
            if r["use"] and r["last_note"] > rec["duration"] + AUDIO_SLACK:
                r.update(use=False, reason=f"audio ends at {rec['duration']:.1f}s, last note at {r['last_note']:.1f}s")
        rec["bpm_test"] = bpm_test(y, song, recs) if do_bpm else None

        tok, meta, k = [], [], 0
        for r in recs:
            m_ = {key: v for key, v in r.items() if key not in ("chart", "view", "tokens")}
            if "tokens" in r:
                tok.append(np.asarray(r["tokens"], dtype=np.uint16))
                m_["token_start"], m_["token_count"] = k, len(r["tokens"])
                k += len(r["tokens"])
            meta.append(m_)
        np.savez(out / "songs" / f"{sid}.npz", mel=m,
                 tokens=np.concatenate(tok) if tok else np.zeros(0, np.uint16),
                 first=np.float64(song["first"]), meta=json.dumps(meta, ensure_ascii=False))
        rec["charts"] = meta
        rec["ok"] = True
    except Exception as e:
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
    return rec


# ---------------------------------------------------------------- summary

def summarize(out: Path) -> dict:
    index = [json.loads(l) for l in (out / "index.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    latest = {}
    for r in index:                                    # a song built twice: the last record wins
        latest[r["id"]] = r
    songs = sorted(latest.values(), key=lambda r: (r["version"], r["song"]))
    good = [s for s in songs if s.get("ok")]

    # usual sync offset, from charts with a clear answer
    def clear(c):
        return c.get("offset_ms") is not None and c.get("sharpness", 0) >= SYNC_SHARP and not c.get("edge")

    usual = float(np.median([c["offset_ms"] for s in good for c in s["charts"] if clear(c)] or [0.0]))
    unclear = []
    seen, rows = {}, []
    for s in good:
        offs = [c["offset_ms"] for c in s["charts"] if clear(c)]
        shift = 0.0
        if offs:
            song_off = float(np.median(offs))
            if abs(song_off - usual) > SYNC_LIMIT_MS:
                if max(abs(o - song_off) for o in offs) <= SYNC_AGREE_MS:
                    shift = round(song_off - usual, 1)               # move the notes onto the audio's hits
                else:
                    unclear.append({"song": f"{s['version']}/{s['song']}", "offsets_ms": offs})
        for c in s["charts"]:
            c = dict(c, shift_ms=shift if c["use"] else 0.0)
            if c["use"] and "hash" in c:                    # the same song's chart copied into another version
                key = (s["title_key"], c["hash"])
                if key in seen:
                    c.update(use=False, reason=f"same chart as {seen[key]}")
                else:
                    seen[key] = f"{s['version']}/{s['song']} slot {c['slot']}"
            rows.append({"id": s["id"], "version": s["version"], "song": s["song"], "split": s["split"],
                         "title_key": s["title_key"], **c})
    with open(out / "charts.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    used = [r for r in rows if r["use"]]
    reasons = Counter(r["reason"].split(" (")[0].split(":")[0] for r in rows if not r["use"])
    bt = [s["bpm_test"] for s in good if s.get("bpm_test")]
    rel = Counter(b["relation"] for b in bt)
    ph = np.array([abs(b["phase_ms"]) for b in bt if b["phase_ms"] is not None])
    offs = np.array([r["offset_ms"] - r["shift_ms"] for r in used if clear(r)])
    tok = np.array([r["token_count"] for r in used])
    report = {
        "songs": len(songs), "songs_failed": [{"song": f"{s['version']}/{s['song']}", "error": s.get("error")}
                                              for s in songs if not s.get("ok")],
        "charts": len(rows), "charts_used": len(used), "not_used": dict(reasons),
        "used_by_split": dict(Counter(r["split"] for r in used)),
        "songs_by_split": dict(Counter(s["split"] for s in good)),
        "used_by_era": dict(Counter(r["era"] for r in used)),
        "used_by_source": dict(Counter(r["source"] for r in used)),
        "used_by_level": dict(sorted(Counter(r["level"] for r in used).items())),
        "split_titles_shared": len({s["title_key"] for s in good if s["split"] == "train"}
                                   & {s["title_key"] for s in good if s["split"] != "train"}),
        "sync": {"usual_offset_ms": round(usual, 1), "charts_judged": int(len(offs)),
                 "within_10ms": round(float(np.mean(np.abs(offs - usual) <= 10)), 3) if len(offs) else None,
                 "within_20ms": round(float(np.mean(np.abs(offs - usual) <= 20)), 3) if len(offs) else None},
        "bpm_test": {"songs": len(bt), "relation": dict(rel),
                     "phase_within_5ms": round(float(np.mean(ph <= 5)), 3) if len(ph) else None,
                     "phase_within_15ms": round(float(np.mean(ph <= 15)), 3) if len(ph) else None},
        "tokens_per_chart": {"median": int(np.median(tok)) if len(tok) else 0,
                             "p99": int(np.percentile(tok, 99)) if len(tok) else 0},
        "sync_unclear": unclear,
        "resynced": sorted(({"chart": f"{r['version']}/{r['song']} slot {r['slot']}", "shift_ms": r["shift_ms"]}
                            for r in rows if r["shift_ms"]), key=lambda x: -abs(x["shift_ms"])),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def verify(out: Path, log=print) -> int:
    """Reopen every song file and check it against charts.jsonl. -> problems found."""
    rows = [json.loads(l) for l in (out / "charts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_song = defaultdict(list)
    for r in rows:
        if "token_start" in r:
            by_song[r["id"]].append(r)
    bad = 0
    for k, (sid, rs) in enumerate(sorted(by_song.items())):
        try:
            with np.load(out / "songs" / f"{sid}.npz") as z:
                m, tok = z["mel"], z["tokens"]
                assert m.ndim == 2 and m.shape[1] == audio.N_MELS and len(m) > 100
                for r in rs:
                    notes = decode(from_ids(tok[r["token_start"]:r["token_start"] + r["token_count"]]))
                    assert len(notes) == r["notes"], f"slot {r['slot']}: {len(notes)} notes, expected {r['notes']}"
        except Exception as e:
            bad += 1
            log(f"  ! {sid} ({rs[0]['version']}/{rs[0]['song']}): {type(e).__name__}: {e}")
        if (k + 1) % 300 == 0:
            log(f"  verified {k + 1}/{len(by_song)} songs", flush=True)
    return bad


def pictures(out: Path, n: int = 6, seconds: float = 12.0) -> None:
    """A few charts' notes drawn over their spectrogram (to check by eye)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: no pictures)")
        return
    rows = [json.loads(l) for l in (out / "charts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    pick = [r for r in rows if r["use"] and r["slot"] in (4, 5)]
    rng = np.random.default_rng(7)
    (out / "pictures").mkdir(exist_ok=True)
    from fractions import Fraction
    from backend.chartcodec.notes import ChartData
    for r in rng.choice(pick, size=min(n, len(pick)), replace=False):
        with np.load(out / "songs" / f"{r['id']}.npz") as z:
            m = z["mel"].astype(np.float32)
            first = float(z["first"])
            notes = decode(from_ids(z["tokens"][r["token_start"]:r["token_start"] + r["token_count"]]))
        cd = ChartData([(Fraction(b), v) for b, v in r["tempo"]], notes)
        times = np.array(sorted({first + cd.seconds(x.beat) for x in notes}))
        mid = times[len(times) // 2]
        t0, t1 = mid - seconds / 2, mid + seconds / 2
        f0, f1 = max(0, int(t0 * audio.FPS)), int(t1 * audio.FPS)
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.imshow(m[f0:f1].T, origin="lower", aspect="auto", cmap="magma",
                  extent=(f0 / audio.FPS, f1 / audio.FPS, 0, audio.N_MELS))
        for t in times[(times >= t0) & (times <= t1)]:
            ax.axvline(t, color="cyan", lw=0.8, alpha=0.8)
        ax.set_title(f"{r['version']}/{r['song']} slot {r['slot']} ({r['level']}): notes (cyan) over the audio")
        ax.set_xlabel("seconds")
        name = f"{r['id']}_{r['slot']}.png"
        fig.savefig(out / "pictures" / name, dpi=80, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--retime", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-bpm-test", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N songs (for a test run)")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    out = args.out.expanduser()
    (out / "songs").mkdir(parents=True, exist_ok=True)

    lock = out / ".lock"
    if lock.exists():
        try:
            pid = int(lock.read_text())
            os.kill(pid, 0)
            sys.exit(f"another build (pid {pid}) is running on {out}; stop it first")
        except (ValueError, ProcessLookupError, PermissionError):
            pass                                       # stale lock from a build that stopped
    lock.write_text(str(os.getpid()))
    try:
        if not args.summary_only:
            build(args, out)
        print("\nsummary ...", flush=True)
        rep = summarize(out)
        print(f"  songs {rep['songs']} ({len(rep['songs_failed'])} failed), charts {rep['charts']}, "
              f"used {rep['charts_used']}")
        print(f"  not used: {rep['not_used']}")
        print(f"  used by split {rep['used_by_split']}, songs by split {rep['songs_by_split']}, "
              f"titles shared between train and val/test: {rep['split_titles_shared']}")
        print(f"  used by era {rep['used_by_era']}, by source {rep['used_by_source']}")
        print(f"  sync: usual offset {rep['sync']['usual_offset_ms']} ms; of {rep['sync']['charts_judged']} "
              f"clear charts, {rep['sync']['within_10ms']} within 10 ms, {rep['sync']['within_20ms']} within 20 ms; "
              f"{len(rep['resynced'])} charts resynced, {len(rep['sync_unclear'])} songs unclear (listed in the report)")
        print(f"  BPM detector: {rep['bpm_test']}")
        print(f"  tokens per chart: {rep['tokens_per_chart']}")
        for f in rep["songs_failed"][:10]:
            print(f"  ! {f['song']}: {f['error']}")
        print("\nverify ...", flush=True)
        bad = verify(out)
        print(f"  {'all song files check out' if not bad else f'{bad} song files have problems (listed above)'}")
        pictures(out)
        print(f"\ndone: {out}/report.json, charts.jsonl, pictures/")
    finally:
        lock.unlink(missing_ok=True)


def build(args, out: Path) -> None:
    collection = args.collection.expanduser()
    songs = find_songs(collection)
    if args.limit:
        songs = songs[:args.limit]
    index_f = out / "index.jsonl"
    done = set()
    if index_f.exists():
        for l in index_f.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("ok") and r.get("format") == FORMAT and (out / "songs" / f"{r['id']}.npz").exists():
                    done.add(r["id"])
    todo = [f for f in songs if song_id(f.parent.relative_to(collection).as_posix()) not in done]
    print(f"{len(songs)} songs, {len(songs) - len(todo)} already built, {len(todo)} to go "
          f"({args.workers} workers)", flush=True)
    jobs = [(f, collection, out, args.retime.expanduser() if args.retime else None, not args.no_bpm_test)
            for f in todo]
    t0, failed = time.time(), 0
    with mp.Pool(args.workers) as pool, open(index_f, "a", encoding="utf-8") as fh:
        for k, rec in enumerate(pool.imap_unordered(build_song, jobs)):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if not rec.get("ok"):
                failed += 1
                print(f"  ! {rec['version']}/{rec['song']}: {rec.get('error')}", flush=True)
            if (k + 1) % 25 == 0 or k + 1 == len(jobs):
                rate = (time.time() - t0) / (k + 1)
                print(f"  {k + 1}/{len(jobs)} songs ({failed} failed), "
                      f"about {rate * (len(jobs) - k - 1) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
