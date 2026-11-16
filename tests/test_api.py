"""API tests via FastAPI's TestClient (needs httpx). Generates audio with ffmpeg."""
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from backend import config
from backend.app import app

pytestmark = pytest.mark.skipif(shutil.which(config.FFMPEG) is None, reason="ffmpeg missing")


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECTS_DIR", tmp_path / "projects")


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mp3(tmp_path):
    p = tmp_path / "track.mp3"
    subprocess.run([config.FFMPEG, "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=3", str(p)], check=True)
    return p


def upload(client, mp3, **form):
    with mp3.open("rb") as f:
        return client.post("/api/projects", files={"audio": ("track.mp3", f, "audio/mpeg")}, data=form)


def test_upload_get_patch_delete(client, mp3):
    r = upload(client, mp3, title="RONDØ", artist="Ashrount")
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["title"] == "RONDØ" and p["artist"] == "Ashrount"
    pid = p["id"]

    assert client.get(f"/api/projects/{pid}").json()["id"] == pid
    assert [x["id"] for x in client.get("/api/projects").json()] == [pid]

    r = client.patch(f"/api/projects/{pid}", json={"title": "New"})
    assert r.json()["title"] == "New" and r.json()["artist"] == "Ashrount"
    assert client.patch(f"/api/projects/{pid}", json={"title": "  "}).status_code == 400

    assert client.delete(f"/api/projects/{pid}").status_code == 200
    assert client.get(f"/api/projects/{pid}").status_code == 404


def test_upload_without_title_uses_filename_not_temp_folder(client, mp3):
    p = upload(client, mp3).json()
    assert p["title"] == "track"  # generic name: user edits it in the UI


def test_audio_routes(client, mp3):
    pid = upload(client, mp3).json()["id"]
    r = client.get(f"/api/projects/{pid}/track.mp3")
    assert r.status_code == 200 and r.content == mp3.read_bytes()
    meta = client.get(f"/api/projects/{pid}/peaks.json").json()
    peaks = client.get(f"/api/projects/{pid}/peaks").content
    assert len(peaks) == meta["count"] * 2
    assert client.get(f"/api/projects/{pid}/jacket").status_code == 404


def test_bad_uploads(client, tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("hi")
    with bad.open("rb") as f:
        assert client.post("/api/projects", files={"audio": ("x.txt", f)}).status_code == 400
    fake = tmp_path / "fake.mp3"
    fake.write_bytes(b"not audio at all")
    with fake.open("rb") as f:
        assert client.post("/api/projects", files={"audio": ("fake.mp3", f)}).status_code == 400
    assert client.get("/api/projects/nope").status_code == 404
    assert client.get("/api/projects/..%2Fetc").status_code == 404


def test_frontend_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "AstroDX autocharter" in r.text
    assert r.headers["cache-control"] == "no-store"
    assert client.get("/js/main.js").status_code == 200


def test_timing_save_and_validate(client, mp3):
    pid = upload(client, mp3).json()["id"]
    assert client.get(f"/api/projects/{pid}").json()["timing"] is None
    r = client.put(f"/api/projects/{pid}/timing", json={"bpm": 314, "first": 0.1234})
    assert r.status_code == 200
    t = client.get(f"/api/projects/{pid}").json()["timing"]
    assert t == {"bpm": 314.0, "first": 0.1234, "confirmed": True}
    for bad in ({"bpm": 0, "first": 0}, {"bpm": 120, "first": -1}, {"bpm": 120, "first": 99}):
        assert client.put(f"/api/projects/{pid}/timing", json=bad).status_code in (400, 422)


def test_detect_timing_route(client, tmp_path):
    from tools.make_test_song import make, write_mp3
    mp3 = tmp_path / "d.mp3"
    write_mp3(make(150, 1.0, 24, seed=2), mp3)
    pid = upload(client, mp3).json()["id"]
    r = client.post(f"/api/projects/{pid}/detect-timing")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["bpm"] == 150 and abs(d["first"] - 1.0) < 0.005
    assert d["candidates"][0]["relation"] == "chosen"
    # cached, and the project's own timing is untouched
    assert client.post(f"/api/projects/{pid}/detect-timing").json() == d
    assert client.get(f"/api/projects/{pid}").json()["timing"] is None