"""Per-project files the music bar needs: the track, waveform peaks, jacket."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend import storage
from backend.api.projects import get_project_or_404

router = APIRouter(tags=["audio"])


@router.get("/projects/{pid}/track.mp3")
def track(pid: str):
    get_project_or_404(pid)
    return FileResponse(storage.project_dir(pid) / "track.mp3", media_type="audio/mpeg")


@router.get("/projects/{pid}/peaks")
def peaks(pid: str):
    """int8 pairs [min, max, ...]; see audio/waveform.py."""
    get_project_or_404(pid)
    return FileResponse(storage.project_dir(pid) / "peaks.bin",
                        media_type="application/octet-stream")


@router.get("/projects/{pid}/peaks.json")
def peaks_meta(pid: str) -> dict:
    get_project_or_404(pid)
    return json.loads((storage.project_dir(pid) / "peaks.json").read_text(encoding="utf-8"))


@router.get("/projects/{pid}/jacket")
def jacket(pid: str):
    p = get_project_or_404(pid)
    if not p.jacket:
        raise HTTPException(404, "no jacket")
    return FileResponse(storage.project_dir(pid) / p.jacket)