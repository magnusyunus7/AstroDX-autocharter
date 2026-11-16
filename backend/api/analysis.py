"""Analysis routes. Phase 1.4: automatic BPM + offset detection."""
from __future__ import annotations

import json

from fastapi import APIRouter

from backend import storage
from backend.api.projects import get_project_or_404
from backend.audio import loader, tempo

router = APIRouter(tags=["analysis"])


def _analysis_path(pid: str):
    return storage.project_dir(pid) / "analysis.json"


def load_analysis(pid: str) -> dict:
    f = _analysis_path(pid)
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def save_analysis(pid: str, data: dict) -> None:
    _analysis_path(pid).write_text(json.dumps(data, indent=2), encoding="utf-8")


@router.post("/projects/{pid}/detect-timing")
def detect_timing(pid: str, force: bool = False) -> dict:
    """Detect BPM/offset (takes a few seconds; cached in analysis.json).
    Never touches the project's confirmed timing: the UI decides what to apply."""
    get_project_or_404(pid)
    data = load_analysis(pid)
    if "timing" in data and not force:
        return data["timing"]
    samples, sr = loader.read_wav(storage.ensure_analysis_wav(pid))
    result = tempo.detect(samples, sr).to_dict()
    data["timing"] = result
    save_analysis(pid, data)
    return result