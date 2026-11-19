"""Generate a chart from the confirmed timing + analysis, and export it."""
from __future__ import annotations

import json
import zlib
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import config, storage
from backend.api.analysis import analyze
from backend.api.projects import get_project_or_404
from backend.export.astrodx import export_level, safe_name
from backend.generator.difficulty import LEVELS
from backend.generator.pipeline import generate

router = APIRouter(tags=["generate"])
SLOTS = {4: "Expert", 5: "Master", 6: "Re:Master"}


class GenerateIn(BaseModel):
    level: str = "14"
    slot: int = 5
    seed: Optional[int] = None       # None: a fixed seed per project+level (same chart every time)


def _gen_path(pid: str):
    return storage.project_dir(pid) / "generated.json"


@router.post("/projects/{pid}/generate")
def generate_chart(pid: str, body: GenerateIn) -> dict:
    p = get_project_or_404(pid)
    if body.level not in LEVELS:
        raise HTTPException(400, f"level must be one of {LEVELS}")
    if body.slot not in SLOTS:
        raise HTTPException(400, f"slot must be one of {list(SLOTS)}")
    analysis = analyze(pid)                      # 400 if timing isn't confirmed; cached otherwise
    seed = body.seed if body.seed is not None else zlib.crc32(f"{pid}:{body.level}".encode()) % 100000
    result = generate(analysis, level=body.level, slot=body.slot, seed=seed,
                      overrides=p.section_levels, title=p.title, artist=p.artist,
                      duration=p.duration)
    d = storage.project_dir(pid)
    (d / "maidata.txt").write_text(result["maidata"], encoding="utf-8", newline="\n")
    _gen_path(pid).write_text(json.dumps(result), encoding="utf-8")
    return result


@router.get("/projects/{pid}/generated")
def last_generated(pid: str) -> Optional[dict]:
    """The last generated chart, or null if there isn't one yet."""
    get_project_or_404(pid)
    f = _gen_path(pid)
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


@router.get("/projects/{pid}/export.zip")
def export_zip(pid: str):
    """AstroDX-ready zip: <Title>/maidata.txt + track.mp3 + bg.(jpg|png)."""
    p = get_project_or_404(pid)
    d = storage.project_dir(pid)
    if not (d / "maidata.txt").is_file():
        raise HTTPException(404, "generate a chart first")
    paths = export_level(config.EXPORTS_DIR, p.title, (d / "maidata.txt").read_text(encoding="utf-8"),
                         d / "track.mp3", d / p.jacket if p.jacket else None)
    return FileResponse(paths["zip"], media_type="application/zip",
                        filename=f"{safe_name(p.title)}.zip")