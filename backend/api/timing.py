"""Timing route: save the BPM + offset the user set in the music bar."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend import storage
from backend.api.projects import get_project_or_404
from backend.models.project import Project, Timing

router = APIRouter(tags=["timing"])


class TimingIn(BaseModel):
    bpm: float = Field(gt=10, le=1000)
    first: float = Field(ge=0)   # seconds to a downbeat (any bar's beat 1)
    confirmed: bool = True


@router.put("/projects/{pid}/timing")
def put_timing(pid: str, body: TimingIn) -> Project:
    p = get_project_or_404(pid)
    if body.first >= p.duration:
        raise HTTPException(400, f"offset {body.first:.3f}s is past the end of the song")
    p.timing = Timing(bpm=round(body.bpm, 4), first=round(body.first, 4),
                      confirmed=body.confirmed)
    storage.save_project(p)
    return p