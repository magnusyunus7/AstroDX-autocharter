"""Phase 0 check: build a hand-written chart that uses every v1 note type,
write maidata.txt and package it for AstroDX.

Usage:
    python -m tools.phase0_demo --audio song.mp3 --bpm 180 --first 1.234 --title "Test"

Pick a song you know the BPM of, and set --first to the time (seconds) of a
downbeat (MajdataEdit's offset is fine). Open the output folder in
MajdataEdit first, then import into AstroDX.
"""
from __future__ import annotations

import argparse
from fractions import Fraction as F
from pathlib import Path

from backend.export.astrodx import export_level
from backend.models.chart import Chart, Hold, Slide, Tap, Touch, TouchHold
from backend.simai.maidata import DifficultyChart, MaidataMeta, build_maidata, rebase_first


def demo_chart(bpm: float) -> Chart:
    c = Chart(bpm)
    b = lambda bar, beat=0: F(bar * 4) + F(beat)  # bar is 0-based

    # bar 0: quarter taps, alternating hands (1 = right side, 8 = left side)
    c.add(*(Tap(b(0, i), 1 if i % 2 == 0 else 8) for i in range(4)))
    # bar 1: 8th-note stream around the ring
    c.add(*(Tap(b(1, F(i, 2)), i + 1) for i in range(8)))
    # bar 2: EACH pairs + a 2-beat hold
    c.add(Tap(b(2, 0), 1), Tap(b(2, 0), 8))
    c.add(Hold(b(2, 1), 4, 2))
    c.add(Tap(b(2, 3), 2), Tap(b(2, 3), 7))
    # bar 3: straight slide 1-5 (star at beat 0, moves beat 1->2), break tap
    c.add(Slide(b(3, 0), 1, "-", 5, 1))
    c.add(Tap(b(3, 3), 3, brk=True))
    # bar 4: circle slide all the way around (1 > 1), plus ex taps
    c.add(Slide(b(4, 0), 1, ">", 1, 2))
    c.add(Tap(b(4, 1), 6, ex=True), Tap(b(4, 2), 5, ex=True), Tap(b(4, 3), 4, ex=True))
    # bar 5: touch notes + a firework touch hold
    c.add(Touch(b(5, 0), "B1"), Touch(b(5, F(1, 2)), "B8"), Touch(b(5, 1), "C"))
    c.add(TouchHold(b(5, 2), 2, "C", firework=True))
    # bar 6: 16th trill 4-5 for 2 beats, then triplets on 2-3-2
    c.add(*(Tap(b(6, F(i, 4)), 4 if i % 2 == 0 else 5) for i in range(8)))
    c.add(*(Tap(b(6, 2 + F(i, 3)), [2, 3, 2][i]) for i in range(3)))
    # bar 7: break EACH to finish
    c.add(Tap(b(7, 0), 1, brk=True), Tap(b(7, 0), 8, brk=True))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--bpm", type=float, required=True)
    ap.add_argument("--first", type=float, required=True,
                    help="seconds from audio start to a downbeat")
    ap.add_argument("--title", default="AstroDX autocharter phase0")
    ap.add_argument("--artist", default="")
    ap.add_argument("--jacket", default=None, help="optional .jpg")
    ap.add_argument("--slot", type=int, default=5, help="5 = Master")
    ap.add_argument("--out", default="data/exports")
    args = ap.parse_args()

    chart = demo_chart(args.bpm)
    first, bars = rebase_first(args.first, args.bpm)
    chart = chart.shifted(bars * 4)

    text = build_maidata(
        MaidataMeta(title=args.title, artist=args.artist, first=first),
        [DifficultyChart(slot=args.slot, level="?", chart=chart)],
    )
    paths = export_level(Path(args.out), args.title, text, args.audio, args.jacket)
    print(text)
    for k, v in paths.items():
        print(f"{k}: {v.resolve()}")


if __name__ == "__main__":
    main()