"""Benchmark timing detection against charts whose timing is known-good.

Point it at a folder of level folders (each with maidata.txt + track.mp3):

    python -m tools.eval_timing samples/

For every level it reads the chart's BPM (first "(bpm)" in the highest
difficulty) and &first, runs detection on track.mp3, and reports:
  BPM     exact / octave (x2, /2 ...) / wrong
  phase   beat-position error in ms (how far detected beats are from the
          chart's beat grid; |err| < ~5 ms is excellent)
  bar     whether detected beat 1 matches the chart's bar grid (bars counted
          from &first). Informative only: charts don't always put their bar
          lines on the musical downbeat.
Charts with BPM changes are listed but skipped.
"""
from __future__ import annotations

import argparse
import re
import tempfile
import time
from pathlib import Path

from backend.audio import loader, tempo


def read_chart_timing(maidata: Path) -> dict:
    text = maidata.read_text(encoding="utf-8", errors="replace")
    first = re.search(r"^&first=([-\d.]+)", text, re.M)
    notes = dict(re.findall(r"^&inote_(\d+)=(.*?)(?=^&|\Z)", text, re.M | re.S))
    if not notes:
        raise ValueError("no &inote_N found")
    body = notes[max(notes, key=int)]
    bpms = re.findall(r"\(([\d.]+)\)", body)
    if not bpms:
        wb = re.search(r"^&wholebpm=([\d.]+)", text, re.M)
        bpms = [wb.group(1)] if wb else []
    if not bpms:
        raise ValueError("no BPM found")
    return {"bpm": float(bpms[0]), "first": float(first.group(1)) if first else 0.0,
            "variable": len({round(float(b), 3) for b in bpms}) > 1}


def relation(det: float, true: float) -> str:
    r = det / true
    for v, name in [(1, "exact"), (2, "x2"), (0.5, "/2"), (1.5, "x1.5"), (2 / 3, "x2/3"),
                    (3, "x3"), (1 / 3, "/3"), (4, "x4"), (0.25, "/4")]:
        if abs(r - v) < 0.0005 * v:
            return name
    return "WRONG"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    a = ap.parse_args()
    levels = sorted(p.parent for p in Path(a.folder).rglob("maidata.txt"))
    rows, stats = [], {"exact": 0, "octave": 0, "wrong": 0, "phase": [], "bar": 0}
    print(f"{'level':30} {'chart bpm':>10} {'detected':>10} {'bpm':>6} {'phase ms':>9} {'bar':>4}  conf      time")
    for lvl in levels:
        name = lvl.name[:30]
        audio = next((lvl / n for n in ("track.mp3", "track.ogg", "track.wav") if (lvl / n).exists()), None)
        if audio is None:
            print(f"{name:30}  (no track.mp3)")
            continue
        try:
            truth = read_chart_timing(lvl / "maidata.txt")
        except ValueError as e:
            print(f"{name:30}  ({e})")
            continue
        if truth["variable"]:
            print(f"{name:30} {truth['bpm']:10.3f}  (BPM changes in chart: skipped)")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "a.wav"
            loader.make_analysis_wav(audio, wav)
            y, sr = loader.read_wav(wav)
        t0 = time.time()
        r = tempo.detect(y, sr)
        dt = time.time() - t0

        rel = relation(r.bpm, truth["bpm"])
        beat = 60 / truth["bpm"]
        if rel == "exact":
            stats["exact"] += 1
            ph = ((r.beat_phase - truth["first"]) / beat + 0.5) % 1 - 0.5
            ph_ms = ph * beat * 1000
            stats["phase"].append(abs(ph_ms))
            bar_off = round(((r.first - truth["first"]) / beat)) % 4
            bar_ok = bar_off == 0
            stats["bar"] += bar_ok
            ph_s, bar_s = f"{ph_ms:+9.1f}", ("ok" if bar_ok else f"+{bar_off}b")
        else:
            stats["octave" if rel != "WRONG" else "wrong"] += 1
            ph_s, bar_s = f"{'-':>9}", "-"
        conf = f"t{r.confidence.get('tempo', 0):.2f} d{r.confidence.get('downbeat', 0):.2f}"
        print(f"{name:30} {truth['bpm']:10.3f} {r.bpm:10.3f} {rel:>6} {ph_s} {bar_s:>4}  {conf}  {dt:4.1f}s")

    n = stats["exact"] + stats["octave"] + stats["wrong"]
    if n:
        ph = sorted(stats["phase"])
        med = ph[len(ph) // 2] if ph else float("nan")
        print(f"\n{n} songs: BPM exact {stats['exact']}, right family but octave {stats['octave']}, "
              f"wrong {stats['wrong']}. Median |phase error| {med:.1f} ms (exact-BPM songs). "
              f"Bar matches chart: {stats['bar']}/{stats['exact']}.")


if __name__ == "__main__":
    main()