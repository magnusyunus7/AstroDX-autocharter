"""FastAPI app: JSON API under /api, the frontend served at /."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import config
from backend.api import analysis, audio, projects, timing

app = FastAPI(title="AstroDX autocharter")
app.include_router(projects.router, prefix="/api")
app.include_router(audio.router, prefix="/api")
app.include_router(timing.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")


class NoCacheStatic(StaticFiles):
    """Serve frontend files without browser caching, so edited JS/CSS shows
    up on a normal refresh."""

    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-store"
        return r


# Must be mounted last: it catches every path the API doesn't.
app.mount("/", NoCacheStatic(directory=config.ROOT / "frontend", html=True), name="frontend")