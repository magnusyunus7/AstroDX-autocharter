"""Analysis routes. Phase 1.4: automatic BPM + offset detection."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend import storage
from backend.api.projects import get_project_or_404
from backend.audio import features, loader, tempo

router = APIRouter(tags=["analysis"])


def _analysis_path(pid: str):
    return storage.project_dir(pid) / "analysis.json"


def load_analysis(pid: str) -> dict:
    f = _analysis_path(pid)
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}


def save_analysis(pid: str, data: dict) -> None:
    _analysis_path(pid).write_text(json.dumps(data, indent=2), encoding="utf-8")


@router.post("/projects/{pid}/detect-timing")
def detect_timing(pid: str, force: bool = False, separate: bool = True) -> dict:
    """Detect BPM/offset (takes a few seconds; cached in analysis.json).
    Re-runs when forced, when the detector version changed, or when a
    different mode is asked for. Never touches the project's confirmed
    timing: the UI decides what to apply."""
    get_project_or_404(pid)
    data = load_analysis(pid)
    cached = data.get("timing")
    if (cached and not force and cached.get("version") == tempo.VERSION
            and cached.get("separate", True) == separate):
        return cached
    samples, sr = loader.read_wav(storage.ensure_analysis_wav(pid))
    result = tempo.detect(samples, sr, separate=separate).to_dict()
    data["timing"] = result
    save_analysis(pid, data)
    return result


@router.post("/projects/{pid}/analyze")
def analyze(pid: str, force: bool = False) -> dict:
    """Phase 2 analysis (grid onsets per stem, bar-by-bar grid fit) on the
    project's *confirmed* timing. Cached in features.json; redone when the
    timing or the analysis version changes."""
    p = get_project_or_404(pid)
    if not p.timing or not p.timing.confirmed:
        raise HTTPException(400, "confirm the timing first")
    f = storage.project_dir(pid) / "features.json"
    if f.is_file() and not force:
        cached = json.loads(f.read_text(encoding="utf-8"))
        if (cached.get("version") == features.VERSION and cached.get("bpm") == p.timing.bpm
                and cached.get("first") == p.timing.first):
            return cached
    samples, sr = loader.read_wav(storage.ensure_analysis_wav(pid))
    result = features.analyze(samples, sr, p.timing.bpm, p.timing.first)
    f.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    return result