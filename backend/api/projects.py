"""Project routes: list, upload, read, rename, delete."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from backend import config, storage
from backend.audio.loader import AudioError
from backend.models.project import Project

router = APIRouter(tags=["projects"])


def get_project_or_404(pid: str) -> Project:
    try:
        return storage.load_project(pid)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, f"project {pid} not found") from None


def _save_upload(up: UploadFile, folder: Path, stem: str, allowed: set[str]) -> Path:
    ext = Path(up.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"unsupported file type {ext or '(none)'!r}")
    dst = folder / f"{stem}{ext}"
    with dst.open("wb") as f:
        while chunk := up.file.read(1 << 20):
            f.write(chunk)
    return dst


@router.get("/projects")
def list_projects() -> list[Project]:
    return storage.list_projects()


@router.post("/projects")
def upload_project(
    audio: UploadFile = File(...),
    jacket: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    artist: Optional[str] = Form(None),
) -> Project:
    """Plain `def` on purpose: FastAPI runs it in a worker thread, so the
    ffmpeg work doesn't block the server."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        audio_path = _save_upload(audio, tmp, "upload", config.AUDIO_EXTS)
        jacket_path = (_save_upload(jacket, tmp, "jacket", config.IMAGE_EXTS)
                       if jacket and jacket.filename else None)
        try:
            p = storage.create_project(audio_path, original_name=audio.filename,
                                       jacket_path=jacket_path, guess_from_folder=False)
        except (ValueError, AudioError) as e:
            raise HTTPException(400, str(e)) from None

    if title or artist:
        if title and title.strip():
            p.title = title.strip()
        if artist is not None:
            p.artist = artist.strip()
        storage.save_project(p)
    return p


@router.get("/projects/{pid}")
def get_project(pid: str) -> Project:
    return get_project_or_404(pid)


class ProjectPatch(BaseModel):
    title: Optional[str] = None
    artist: Optional[str] = None


@router.patch("/projects/{pid}")
def patch_project(pid: str, body: ProjectPatch) -> Project:
    p = get_project_or_404(pid)
    if body.title is not None:
        if not body.title.strip():
            raise HTTPException(400, "title can't be empty")
        p.title = body.title.strip()
    if body.artist is not None:
        p.artist = body.artist.strip()
    storage.save_project(p)
    return p


@router.delete("/projects/{pid}")
def delete_project(pid: str) -> dict:
    get_project_or_404(pid)
    storage.delete_project(pid)
    return {"deleted": pid}