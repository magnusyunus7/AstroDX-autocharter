"""Phase C2 (part 2, GPU): MERT features for every song in the dataset.

    python -m ml.mert_features --collection ~/collection --data ~/dataset --check
    python -m ml.mert_features --collection ~/collection --data ~/dataset

MERT (m-a-p/MERT-v1-95M) is an open model trained on a lot of music: its
features know about beats, instruments, sections and repetition, which our
small chart model can't learn from ~1,800 songs alone. Licence: CC BY-NC 4.0
(non-commercial use only).

Per song: the audio decoded at 24 kHz (MERT's rate, same ffmpeg as the rest),
run through MERT in 20 s pieces with 2.5 s of context on each side, every
layer's output averaged (75 frames per second, 768 numbers each), then
shrunk to 256 numbers per frame with PCA fitted on training songs. Saved as
<data>/mert/<id>.npy (float16, frames x 256); frame k is the moment k / 75 s.

--check runs one song and prints what it got, before the long run.
Safe to stop and run again: finished songs are skipped. Needs a GPU, the
transformers package and internet access to huggingface.co the first time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml import audio                                                    # noqa: E402

MODEL = "m-a-p/MERT-v1-95M"
SR = 24000
FPS = 75
DIM = 256
CHUNK, CONTEXT = 20.0, 2.5            # seconds
PCA_SONGS, PCA_FRAMES = 150, 400      # songs and frames per song used to fit the PCA


def load_model(device):
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True).to(device).eval()
    proc = Wav2Vec2FeatureExtractor.from_pretrained(MODEL, trust_remote_code=True)
    if device == "cuda":
        model = model.half()
    return model, proc, torch


def embed(y: np.ndarray, model, proc, torch, device) -> np.ndarray:
    """Mono 24 kHz samples -> (frames, 768) float32, every layer averaged."""
    n_out = int(len(y) / SR * FPS)
    out = np.zeros((n_out, 768), np.float32)
    step, ctx = int(CHUNK * SR), int(CONTEXT * SR)
    for s in range(0, len(y), step):
        a, b = max(0, s - ctx), min(len(y), s + step + ctx)
        x = proc(y[a:b], sampling_rate=SR, return_tensors="pt")["input_values"].to(device)
        if device == "cuda":
            x = x.half()
        with torch.no_grad():
            hs = model(x, output_hidden_states=True).hidden_states
        h = torch.stack(hs).mean(0)[0].float().cpu().numpy()     # (frames, 768)
        f_a = int(round(a / SR * FPS))
        lo = int(round(s / SR * FPS)) - f_a                         # drop the context frames
        hi = min(int(round(min(len(y), s + step) / SR * FPS)) - f_a, len(h))
        dst = int(round(s / SR * FPS))
        k = max(0, min(hi - lo, n_out - dst))
        out[dst:dst + k] = h[lo:lo + k]
    return out


def songs_of(data: Path) -> list[dict]:
    latest = {}
    for l in (data / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            latest[r["id"]] = r
    return [r for r in latest.values() if r.get("ok")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    data, coll = args.data.expanduser(), args.collection.expanduser()
    (data / "mert").mkdir(exist_ok=True)
    import torch as _t
    device = "cuda" if _t.cuda.is_available() else "cpu"
    songs = songs_of(data)
    path_of = lambda r: coll / r["version"] / r["song"] / r["audio"]
    print(f"{len(songs)} songs, device {device}; loading {MODEL} ...", flush=True)
    model, proc, torch = load_model(device)

    if args.check:
        r = songs[0]
        t0 = time.time()
        y = audio.decode(path_of(r), sr=SR)
        e = embed(y, model, proc, torch, device)
        print(f"{r['version']}/{r['song']}: {len(y) / SR:.1f} s of audio -> {e.shape} features in "
              f"{time.time() - t0:.1f} s (expected about {len(y) / SR * FPS:.0f} x 768); "
              f"values {e.mean():.3f} +- {e.std():.3f}, NaN: {bool(np.isnan(e).any())}")
        return

    lock = data / ".mert.lock"
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            sys.exit("another MERT run is going on this folder; stop it first")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    lock.write_text(str(os.getpid()))
    try:
        pca_f = data / "mert_pca.npz"
        if not pca_f.exists():
            rng = np.random.default_rng(0)
            train = [r for r in songs if r["split"] == "train"]
            pick = rng.choice(len(train), size=min(PCA_SONGS, len(train)), replace=False)
            rows = []
            for k, i in enumerate(pick):
                e = embed(audio.decode(path_of(train[i]), sr=SR), model, proc, torch, device)
                rows.append(e[rng.choice(len(e), size=min(PCA_FRAMES, len(e)), replace=False)])
                if (k + 1) % 25 == 0:
                    print(f"  PCA: {k + 1}/{len(pick)} songs", flush=True)
            X = np.concatenate(rows)
            mean = X.mean(0)
            _, sv, vt = np.linalg.svd(X - mean, full_matrices=False)
            kept = float((sv[:DIM] ** 2).sum() / (sv ** 2).sum())
            np.savez(pca_f, mean=mean, components=vt[:DIM], kept=kept)
            print(f"  PCA fitted on {len(X)} frames: {DIM} numbers keep {kept:.1%} of the variation", flush=True)
        with np.load(pca_f) as z:
            mean, comp = z["mean"], z["components"]
        todo = [r for r in songs if not (data / "mert" / f"{r['id']}.npy").exists()]
        print(f"{len(songs) - len(todo)} songs already done, {len(todo)} to go", flush=True)
        t0 = time.time()
        for k, r in enumerate(todo):
            try:
                e = embed(audio.decode(path_of(r), sr=SR), model, proc, torch, device)
                z = ((e - mean) @ comp.T).astype(np.float16)
                tmp = data / "mert" / f"{r['id']}.tmp.npy"
                np.save(tmp, z)
                tmp.replace(data / "mert" / f"{r['id']}.npy")
            except Exception as ex:
                print(f"  ! {r['version']}/{r['song']}: {type(ex).__name__}: {ex}", flush=True)
            if (k + 1) % 25 == 0 or k + 1 == len(todo):
                rate = (time.time() - t0) / (k + 1)
                print(f"  {k + 1}/{len(todo)} songs, about {rate * (len(todo) - k - 1) / 60:.0f} min left", flush=True)
        missing = [r for r in songs if not (data / "mert" / f"{r['id']}.npy").exists()]
        print(f"done: {len(songs) - len(missing)} of {len(songs)} songs have MERT features"
              + (f"; missing: {[r['song'] for r in missing[:10]]}" if missing else ""))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
