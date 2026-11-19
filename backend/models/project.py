"""Project metadata, saved as data/projects/<id>/project.json."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class Timing(BaseModel):
    bpm: float
    first: float = 0.0          # seconds from audio start to bar 1 beat 1
    confirmed: bool = False     # True once the user locks it in the music bar


class Project(BaseModel):
    id: str
    created: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_name: str            # original upload filename
    title: str
    artist: str = ""
    duration: float             # seconds, from the decoded audio
    jacket: Optional[str] = None  # "bg.jpg" / "bg.png" inside the project folder
    timing: Optional[Timing] = None
    # user overrides of section levels, keyed by the section's start bar
    # (analysis bar index as a string). Ignored if no section starts there.
    section_levels: dict[str, str] = Field(default_factory=dict)