"""Package a level for AstroDX:  <out_dir>/<Title>/{maidata.txt, track.mp3, bg.jpg}
plus a .zip of that folder.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path


def safe_name(s: str, fallback: str = "untitled") -> str:
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", s).strip(" .")
    return s[:80] or fallback


def export_level(out_dir: Path | str, title: str, maidata_text: str,
                 track_mp3: Path | str, jacket: Path | str | None = None,
                 make_zip: bool = True) -> dict[str, Path]:
    out_dir = Path(out_dir)
    track_mp3 = Path(track_mp3)
    if track_mp3.suffix.lower() != ".mp3":
        raise ValueError("track must be an .mp3 (convert it first)")
    if not track_mp3.is_file():
        raise FileNotFoundError(track_mp3)

    level_dir = out_dir / safe_name(title)
    if level_dir.exists():
        shutil.rmtree(level_dir)
    level_dir.mkdir(parents=True)

    # AstroDX refuses imports with stray files, so write exactly these.
    (level_dir / "maidata.txt").write_text(maidata_text, encoding="utf-8", newline="\n")
    shutil.copyfile(track_mp3, level_dir / "track.mp3")
    if jacket:
        jacket = Path(jacket)
        ext = jacket.suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            raise ValueError("jacket must be .jpg or .png")
        if not jacket.is_file():
            raise FileNotFoundError(jacket)
        # keep the original format: bg.jpg or bg.png
        shutil.copyfile(jacket, level_dir / ("bg.png" if ext == ".png" else "bg.jpg"))

    result = {"folder": level_dir}
    if make_zip:
        zip_path = out_dir / f"{level_dir.name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(level_dir.iterdir()):
                z.write(f, arcname=f"{level_dir.name}/{f.name}")
        result["zip"] = zip_path
    return result