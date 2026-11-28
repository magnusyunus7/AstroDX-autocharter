"""Training examples for the chart model (Phase C3).

One example = one chart (or a window of it, for long charts):
  encoder input  the song cut into quarter-beat steps along the chart's tempo
                 map. Each step: the mel spectrogram at its 8 32nd-note grid
                 points (where notes usually go) and the MERT features in its
                 middle. Step k starts at beat k/4.
  decoder input  the chart's C1 tokens (model view), each with the encoder
                 step it belongs to, so the writer knows which moment of the
                 music it's at.
  settings       level (number), difficulty slot, era, source, BPM.
Training examples are rotated / mirrored at random (16 variants per chart).
"""
from __future__ import annotations

import json
import math
import random
from fractions import Fraction
from pathlib import Path

import numpy as np

from backend.chartcodec.augment import transform
from backend.chartcodec.notes import GRID
from backend.chartcodec.tokens import INDEX, VOCAB, decode, encode, from_ids, ids
from ml.audio import FPS as MEL_FPS, N_MELS

STEPS_PER_BEAT = 4
SUB = 8                                 # mel samples per step
MERT_FPS, MERT_DIM = 75, 256
MEL_DIM = SUB * N_MELS
MAX_TOKENS = 2048                       # decoder window
MAX_STEPS = 4096                        # encoder steps (1,024 beats)
PAD, BOS, EOS, BEAT = INDEX["PAD"], INDEX["BOS"], INDEX["EOS"], INDEX["BEAT"]
SLOTS, ERAS, SOURCES = 7, ("STD", "DX"), ("converted", "proofread")
BPM_BUCKETS = 16


def bpm_bucket(bpm: float) -> int:
    x = (math.log(min(400.0, max(60.0, bpm))) - math.log(60.0)) / (math.log(400.0) - math.log(60.0))
    return min(BPM_BUCKETS - 1, int(x * BPM_BUCKETS))


def level_weight(level: float) -> float:
    """How often a chart is picked for training: 13 and up (the levels we judge
    by) 3x, 10-12.9 2x, below 10 1x."""
    return 3.0 if level >= 13 else 2.0 if level >= 10 else 1.0


# ---------------------------------------------------------------- timing

def beats_to_seconds(tempo: list[tuple[float, float]], beats: np.ndarray) -> np.ndarray:
    """Seconds from beat 0 for many beat positions (tempo: [(beat, bpm), ...])."""
    starts = np.array([b for b, _ in tempo], dtype=np.float64)
    bpms = np.array([v for _, v in tempo], dtype=np.float64)
    sec_at = np.zeros(len(tempo))
    for i in range(1, len(tempo)):
        sec_at[i] = sec_at[i - 1] + (starts[i] - starts[i - 1]) * 60.0 / bpms[i - 1]
    k = np.clip(np.searchsorted(starts, beats, side="right") - 1, 0, len(tempo) - 1)
    return sec_at[k] + (beats - starts[k]) * 60.0 / bpms[k]


def token_beats(tok: list[int]) -> np.ndarray:
    """Beat position (float) of every token: the moment it belongs to."""
    out = np.zeros(len(tok), dtype=np.float64)
    beat_no, pos, i = -1, 0.0, 0
    while i < len(tok):
        t = VOCAB[tok[i]]
        if t == "BEAT":
            beat_no += 1; pos = 0.0
        elif t[0] == "P" and t[1:].isdigit():
            pos = int(t[1:]) / GRID
        elif t[0] == "T" and t[1:].isdigit() and i + 1 < len(tok):
            d = int(t[1:]); n = VOCAB[tok[i + 1]]
            pos = int(n[1:]) / d if n[0] == "N" else pos
            out[i] = max(0, beat_no) + pos
            i += 1
        out[i] = max(0, beat_no) + pos
        i += 1
    return out


# ---------------------------------------------------------------- one example

class SongStore:
    """Loads a song's mel / MERT on demand (keeps the last few in memory)."""

    def __init__(self, data: Path, cache: int = 64):
        self.data, self.cache, self.keep = data, {}, cache

    def get(self, sid: str):
        if sid not in self.cache:
            if len(self.cache) >= self.keep:
                self.cache.pop(next(iter(self.cache)))
            with np.load(self.data / "songs" / f"{sid}.npz") as z:
                mel = z["mel"]
                tokens = z["tokens"]
                first = float(z["first"])
            mf = self.data / "mert" / f"{sid}.npy"
            mert = np.load(mf) if mf.exists() else None
            self.cache[sid] = (mel, mert, tokens, first)
        return self.cache[sid]


def encoder_features(mel, mert, first: float, shift_s: float, tempo, n_steps: int, features: str,
                     s0: int = 0) -> np.ndarray:
    """(n_steps, MEL_DIM + MERT_DIM) float16 for steps s0 .. s0+n_steps-1."""
    k = np.arange(s0, s0 + n_steps, dtype=np.float64)
    sub = (k[:, None] + np.arange(SUB)[None, :] / SUB) / STEPS_PER_BEAT        # beats: the 32nd-note grid points
    t_sub = first + shift_s + beats_to_seconds(tempo, sub.ravel()).reshape(n_steps, SUB)
    out = np.zeros((n_steps, MEL_DIM + MERT_DIM), dtype=np.float32)
    if features in ("mel", "both"):
        fi = np.round(t_sub * MEL_FPS).astype(np.int64)
        ok = (fi >= 0) & (fi < len(mel))
        m = (mel[np.clip(fi, 0, len(mel) - 1)].astype(np.float32) + 40.0) / 40.0       # dB -> about -1..1
        m[~ok] = -1.0                                                                  # silence
        out[:, :MEL_DIM] = m.reshape(n_steps, MEL_DIM)
    if features in ("mert", "both") and mert is not None:
        tc = first + shift_s + beats_to_seconds(tempo, (k + 0.5) / STEPS_PER_BEAT)
        fi = np.round(tc * MERT_FPS).astype(np.int64)
        ok = (fi >= 0) & (fi < len(mert))
        out[:, MEL_DIM:] = np.where(ok[:, None], mert[np.clip(fi, 0, len(mert) - 1)].astype(np.float32) / 4.0, 0.0)
    return out.astype(np.float16)


def hide_bars(tok_in: np.ndarray, beats: np.ndarray, p: float, rng: random.Random) -> np.ndarray:
    """Training only: blank out (PAD) the notes of a random share p of bars in
    the decoder's input, keeping BEAT / BOS markers. The model then can't just
    continue its own recent pattern and has to take more from the audio.
    The targets stay complete: it still learns to write every note."""
    if p <= 0:
        return tok_in
    out = tok_in.copy()
    bars = np.floor(beats[:len(out)] / 4).astype(np.int64)
    hide = {b for b in np.unique(bars) if rng.random() < p}
    keep = (out == BEAT) | (out == BOS)
    m = np.isin(bars, list(hide)) & ~keep
    out[m] = PAD
    return out


def make_example(row: dict, store: SongStore, features: str, rng: random.Random | None,
                 augment: bool, max_tokens: int = MAX_TOKENS, start: int | None = None,
                 hide: float = 0.0) -> dict:
    """One training example from a charts.jsonl row. rng None = deterministic
    (validation: no augmentation, window from the start)."""
    mel, mert, all_tok, first = store.get(row["id"])
    tok = [int(x) for x in all_tok[row["token_start"]:row["token_start"] + row["token_count"]]]
    if augment and rng is not None:
        r, m = rng.randrange(8), rng.random() < 0.5
        if r or m:
            tok = ids(encode(transform(decode(from_ids(tok)), r, m)))
    beats = token_beats(tok)
    # window: whole chart if it fits, otherwise a random stretch starting at a BEAT
    if len(tok) > max_tokens + 1:
        if start is None:
            starts = [i for i, t in enumerate(tok) if t == BEAT and i + max_tokens + 1 <= len(tok)] or [0]
            start = rng.choice(starts) if rng is not None else 0
        tok, beats = tok[start:start + max_tokens + 1], beats[start:start + max_tokens + 1]
    tempo = [(float(Fraction(b)), v) for b, v in row["tempo"]]
    step = np.floor(beats * STEPS_PER_BEAT).astype(np.int64)
    s0 = max(0, int(step[-1]) + 16 - MAX_STEPS)                       # encoder crop for very long songs
    n_steps = int(step[-1]) + 16 - s0
    enc = encoder_features(mel, mert, first, row.get("shift_ms", 0.0) / 1000.0, tempo, n_steps, features, s0)
    return {
        "enc": enc, "bar": (np.arange(s0, s0 + n_steps) % (4 * STEPS_PER_BEAT)).astype(np.int64),
        "tok_in": hide_bars(np.array(tok[:-1], np.int64), beats, hide, rng) if (hide and rng is not None)
                  else np.array(tok[:-1], np.int64),
        "tok_out": np.array(tok[1:], np.int64),
        "tok_step": np.clip(step[:-1] - s0, 0, n_steps - 1),
        "level": np.float32(row["level_value"]), "slot": row["slot"],
        "era": ERAS.index(row["era"]), "source": SOURCES.index(row["source"]),
        "bpm": bpm_bucket(tempo[0][1]),
    }


def collate(batch: list[dict]) -> dict:
    """Examples -> one padded batch of numpy arrays (no torch here: worker
    processes send these through a pipe; torch tensors would need shared
    memory, which cluster containers often limit). to_torch() converts."""
    B = len(batch)
    S = max(len(b["enc"]) for b in batch)
    T = max(len(b["tok_in"]) for b in batch)
    enc = np.zeros((B, S, MEL_DIM + MERT_DIM), np.float16)
    enc_mask = np.ones((B, S), bool)                                  # True = padding
    bar = np.zeros((B, S), np.int64)
    tin = np.full((B, T), PAD, np.int64)
    tout = np.full((B, T), PAD, np.int64)
    tstep = np.zeros((B, T), np.int64)
    for i, b in enumerate(batch):
        s, t = len(b["enc"]), len(b["tok_in"])
        enc[i, :s] = b["enc"]; enc_mask[i, :s] = False; bar[i, :s] = b["bar"]
        tin[i, :t] = b["tok_in"]; tout[i, :t] = b["tok_out"]; tstep[i, :t] = b["tok_step"]
    cond = {"level": np.array([b["level"] for b in batch], np.float32),
            **{k: np.array([b[k] for b in batch], np.int64) for k in ("slot", "era", "source", "bpm")}}
    return {"enc": enc, "enc_mask": enc_mask, "bar": bar, "tok_in": tin, "tok_out": tout, "tok_step": tstep, **cond}


def to_torch(batch: dict, device=None) -> dict:
    import torch
    return {k: torch.from_numpy(v).to(device, non_blocking=True) if device else torch.from_numpy(v)
            for k, v in batch.items()}


def load_rows(data: Path, split: str) -> list[dict]:
    rows = [json.loads(l) for l in (data / "charts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["use"] and r["split"] == split and "token_start" in r]
