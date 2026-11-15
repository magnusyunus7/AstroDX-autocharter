"""Start Astrodx autocharter:  python run.py  [--port 8000] [--no-browser] [--reload]

Opens http://127.0.0.1:<port>/ in your browser. Only listens on localhost,
so nothing is reachable from other machines.
"""
from __future__ import annotations

import argparse
import threading
import webbrowser

import uvicorn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--reload", action="store_true",
                    help="restart the server when backend .py files change")
    a = ap.parse_args()

    url = f"http://127.0.0.1:{a.port}/"
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"AstroDX autocharter running at {url}  (Ctrl+C to stop)")
    uvicorn.run("backend.app:app", host="127.0.0.1", port=a.port,
                reload=a.reload, reload_dirs=["backend"] if a.reload else None,
                log_level="warning")


if __name__ == "__main__":
    main()