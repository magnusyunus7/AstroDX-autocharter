"""Zip every maidata.txt of the chart collection (no audio, no videos) so the
charts can be scanned elsewhere. Only the small text files are read, so a
streamed Google Drive folder downloads almost nothing.

    python -m tools.pack_maidata "G:\\My Drive\\maisquared"

Skips utage folders ("[..."), "02.5. 舞萌中国", "25A. KALEIDXSCOPE Playlists"
and "legacy_maidata (DO NOT DOWNLOAD)". Writes data/maidata_all.zip, with
paths like "24. PRiSM/RONDØ/maidata.txt".
"""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = ("02.5. 舞萌中国", "25A. KALEIDXSCOPE Playlists", "legacy_maidata (DO NOT DOWNLOAD)")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1]).expanduser()
    if not src.is_dir():
        sys.exit(f"not a folder: {src}")
    out = ROOT / "data" / "maidata_all.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.time()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for version in sorted(p for p in src.iterdir() if p.is_dir()):
            if version.name in EXCLUDE:
                continue
            k = 0
            for dirpath, dirnames, filenames in os.walk(version):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith("["))
                if "maidata.txt" in filenames:
                    f = Path(dirpath) / "maidata.txt"
                    z.write(f, f.relative_to(src).as_posix())
                    n += 1; k += 1
            print(f"  {version.name}: {k} songs ({n} total, {time.time() - t0:.0f} s)", flush=True)
    print(f"done: {n} maidata.txt in {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
