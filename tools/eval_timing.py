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
An empty lead-in at another BPM (e.g. "(172){1}," then constant "(57.333)")
is understood. Charts whose notes change BPM are listed but skipped.
"""
from __future__ import annotations

import argparse
import re
import tempfile
import time
from pathlib import Path

from backend.audio import loader, tempo


def scan_simai(body: str):
    """Walk a simai note string up to the first note. Returns
    (grid_start, bpm_at_first_note, later_bpms): when the BPM the notes use
    was last set (an empty lead-in can use its own BPM, e.g. "(172){1},"
    before "(57.333)"; the real grid starts where that BPM begins), that BPM,
    and every BPM marker after the first note."""
    i, n = 0, len(body)
    bpm, div, t, grid_start = None, 4.0, 0.0, 0.0
    while i < n:
        c = body[i]
        if c == "(":
            j = body.index(")", i)
            new = float(body[i + 1:j]); i = j + 1
            if bpm is None or abs(new - bpm) > 1e-6:
                grid_start = t
            bpm = new
        elif c == "{":
            j = body.index("}", i)
            v = body[i + 1:j]
            if v.startswith("#"):          # {#sec}: each comma lasts sec seconds
                div = -float(v[1:])
            else:
                div = float(v)
            i = j + 1
        elif c == ",":
            if bpm is None:
                raise ValueError("comma before any BPM")
            t += -div if div < 0 else 240.0 / bpm / div
            i += 1
        elif c.isspace():
            i += 1
        else:
            break                          # first note (or E)
    later = [float(x) for x in re.findall(r"\(([\d.]+)\)", body[i:])]
    return grid_start, bpm, later


def read_chart_timing(maidata: Path) -> dict:
    text = maidata.read_text(encoding="utf-8", errors="replace")
    first = re.search(r"^&first=([-\d.]+)", text, re.M)
    first = float(first.group(1)) if first else 0.0
    notes = dict(re.findall(r"^&inote_(\d+)=(.*?)(?=^&|\Z)", text, re.M | re.S))
    if not notes:
        raise ValueError("no &inote_N found")
    body = notes[max(notes, key=int)]
    lead, bpm, later = scan_simai(body)
    if bpm is None:
        wb = re.search(r"^&wholebpm=([\d.]+)", text, re.M)
        if not wb:
            raise ValueError("no BPM found")
        bpm = float(wb.group(1))
    variable = any(abs(b - bpm) > 1e-3 for b in later)
    # the notes' beat grid starts after the lead-in
    return {"bpm": bpm, "first": first + lead, "variable": variable, "lead": lead}


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
    ap.add_argument("--debug", action="store_true",
                    help="also print each BPM candidate's octave ratios (for tuning)")
    ap.add_argument("--fast", action="store_true",
                    help="skip drum/melody separation (~4x faster, slightly less accurate)")
    a = ap.parse_args()
    print(f"mode: {'fast (no separation)' if a.fast else 'separate (default)'}\n")
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
        r = tempo.detect(y, sr, separate=not a.fast)
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
        if a.debug:
            for c in sorted(r.candidates, key=lambda c: c.bpm):
                mark = " <- chart" if abs(c.bpm - truth["bpm"]) < 1e-3 else ""
                print(f"{'':32}{c.bpm:9.3f} {c.relation:>6}  body {c.grid.get('body', 0):.2f}  "
                      f"low {c.grid.get('low', 0):.2f}{mark}")

    n = stats["exact"] + stats["octave"] + stats["wrong"]
    if n:
        ph = sorted(stats["phase"])
        med = ph[len(ph) // 2] if ph else float("nan")
        print(f"\n{n} songs: BPM exact {stats['exact']}, right family but octave {stats['octave']}, "
              f"wrong {stats['wrong']}. Median |phase error| {med:.1f} ms (exact-BPM songs). "
              f"Bar matches chart: {stats['bar']}/{stats['exact']}.")


if __name__ == "__main__":
    main()