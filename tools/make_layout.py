"""Make layout drafts for a song (to try in AstroDX / MajdataEdit).

    python tools/make_layout.py --audio song.mp3 --out drafts/song --bpm 190 --first 0.0
    python tools/make_layout.py --ref samples/IF_U --out drafts/IF_U

--ref: a song folder with track.mp3 + maidata.txt: BPM and &first come from
its Master chart (single-BPM songs only), and that chart is put in as
Re:Master to compare with. Without --bpm, the app's BPM / downbeat detection
is used.

Writes <out>/maidata.txt (light as Advanced, medium as Expert, dense as
Master by default; --densities to change) and copies the track.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.layout import features as F, make_layout, maidata_text      # noqa: E402

SLOT = {"basic": 2, "light": 3, "medium": 4, "dense": 5}


def decode(path: Path):
    import subprocess
    import numpy as np
    import shutil as sh
    exe = sh.which("ffmpeg")
    if not exe:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn", "-ac", "1",
                        "-ar", "22050", "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype="<f4").copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path)
    ap.add_argument("--ref", type=Path, help="song folder with track.mp3 + maidata.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--first", type=float, help="seconds of the first downbeat")
    ap.add_argument("--densities", default="light,medium,dense")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title")
    args = ap.parse_args()

    real = None
    title = args.title
    audio = args.audio
    bpm, first = args.bpm, args.first
    if args.ref:
        from backend.chartcodec.parse import parse_chart, parse_maidata
        audio = audio or args.ref / "track.mp3"
        md = parse_maidata((args.ref / "maidata.txt").read_text(encoding="utf-8"))
        slot = 5 if 5 in md["charts"] else max(md["charts"])
        chart, _ = parse_chart(md["charts"][slot]["body"], md["wholebpm"], repair=True)
        if len(chart.tempo) != 1:
            sys.exit(f"{args.ref}: the reference chart changes BPM; give --bpm and --first instead")
        bpm = bpm or chart.tempo[0][1]
        first = md["first"] if first is None else first
        real = (md["charts"][slot]["level"] or "?", chart.notes)
        title = title or md["title"]
    if audio is None:
        sys.exit("give --audio or --ref")
    y = decode(audio)
    duration = len(y) / 22050
    if bpm is None or first is None:
        from backend.audio import tempo
        res = tempo.detect(y, 22050)
        bpm = bpm or res.bpm
        first = res.first if first is None else first
        print(f"detected: {res.bpm} BPM, first downbeat {res.first:.3f}s")
    mel = F.mel(y)

    charts, report = {}, {}
    for d in args.densities.split(","):
        lay = make_layout(mel, bpm, first, duration, d, seed=args.seed)
        charts[SLOT[d]] = (lay.level, lay.notes)
        report[d] = lay.stats
        s = lay.stats
        print(f"{d:<7} {s['notes']:>4} notes  {s['moments']:>4} moments  on hits {s['on_hits']}  "
              f"density r {s['density_r']}  checker fixes {s['checker_fixes']}  "
              f"mirrored reuse {s['reused_from_mirrored_phrase']}")
    if real:
        charts[6] = real
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "maidata.txt").write_text(maidata_text(title or audio.stem, "", bpm, first, charts),
                                          encoding="utf-8")
    shutil.copyfile(audio, args.out / "track.mp3")
    (args.out / "layout_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"wrote {args.out / 'maidata.txt'}")


if __name__ == "__main__":
    main()
