"""Batch 2 check: import a song into a project folder from the command line.

Usage:
    python -m tools.import_song samples/RONDØ/track.mp3
    python -m tools.import_song song.flac --jacket cover.webp
    python -m tools.import_song --list
    python -m tools.import_song --delete <id>

Writes data/projects/<id>/ and a waveform_preview.svg you can open in a
browser to eyeball the peaks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from backend import storage


def sha1(p: Path) -> str:
    return hashlib.sha1(p.read_bytes()).hexdigest()[:10]


def write_svg_preview(peaks_path: Path, meta: dict, out: Path, width: int = 1600, height: int = 200) -> None:
    peaks = np.frombuffer(peaks_path.read_bytes(), dtype=np.int8).reshape(-1, 2)
    n = len(peaks)
    per_px = max(1, n // width)
    cols = n // per_px
    p = peaks[: cols * per_px].reshape(cols, per_px, 2)
    mn, mx = p[:, :, 0].min(axis=1), p[:, :, 1].max(axis=1)
    mid = height / 2
    scale = (height / 2 - 2) / 127
    lines = [f'<line x1="{x}" y1="{mid - mx[x] * scale:.1f}" x2="{x}" y2="{mid - mn[x] * scale:.1f}"/>'
             for x in range(cols)]
    dur = n * meta["spp"] / meta["sample_rate"]
    ticks = []
    for s in range(0, int(dur) + 1, 10):  # a tick every 10 s
        x = s / dur * cols
        ticks.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" stroke="#555"/>'
                     f'<text x="{x + 3:.1f}" y="12" fill="#aaa" font-size="11">{s}s</text>')
    fs = meta.get("first_sound", 0) / dur * cols
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols}" height="{height}" '
           f'><rect width="100%" height="100%" fill="#111"/>{"".join(ticks)}'
           f'<g stroke="#4fc3f7">{"".join(lines)}</g>'
           f'<line x1="{fs:.1f}" y1="0" x2="{fs:.1f}" y2="{height}" stroke="#ff5252"/></svg>')
    out.write_text(svg, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--jacket")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delete")
    a = ap.parse_args()

    if a.list:
        for p in storage.list_projects():
            print(f"{p.id}  {p.duration:7.2f}s  {p.title} - {p.artist}  jacket={p.jacket}")
        return
    if a.delete:
        storage.delete_project(a.delete)
        print("deleted", a.delete)
        return
    if not a.audio:
        ap.error("give an audio file, --list or --delete")

    src = Path(a.audio)
    p = storage.create_project(src, jacket_path=Path(a.jacket) if a.jacket else None)
    d = storage.project_dir(p.id)
    meta = json.loads((d / "peaks.json").read_text(encoding="utf-8"))
    write_svg_preview(d / "peaks.bin", meta, d / "waveform_preview.svg")

    print(p.model_dump_json(indent=2))
    print(f"peaks:       {meta['count']} pairs ({meta['sample_rate'] / meta['spp']:.0f}/s)")
    print(f"first sound: {meta['first_sound']}s")
    if src.suffix.lower() == ".mp3":
        same = sha1(src) == sha1(d / "track.mp3")
        print(f"track.mp3 identical to upload: {same}")
    print(f"folder:      {d.resolve()}")
    print(f"preview:     {(d / 'waveform_preview.svg').resolve()}")


if __name__ == "__main__":
    main()