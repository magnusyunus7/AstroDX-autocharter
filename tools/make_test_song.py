"""Generate a synthetic song with exactly known BPM and downbeat, for testing
timing detection. Drums (kick on 1 & 3, snare on 2 & 4, offbeat hats), bass +
chords that change on every downbeat, a pickup before bar 1, a drumless
breakdown, and +-3 ms human timing jitter.

Usage:
    python -m tools.make_test_song out.mp3 --bpm 157.3333 --first 1.234 --bars 48
"""
from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from backend import config

SR = 44100


def _env(n, decay):
    return np.exp(-np.arange(n) / (decay * SR))


def kick(sr=SR):
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    f = 50 + 90 * np.exp(-t * 35)                   # pitch drop
    return 0.9 * np.sin(2 * np.pi * np.cumsum(f) / sr) * _env(n, 0.09)


def snare(rng, sr=SR):
    n = int(0.18 * sr)
    t = np.arange(n) / sr
    body = 0.4 * np.sin(2 * np.pi * 190 * t) * _env(n, 0.04)
    return body + 0.5 * rng.standard_normal(n) * _env(n, 0.05)


def hat(rng, sr=SR):
    n = int(0.05 * sr)
    x = rng.standard_normal(n)
    x = np.diff(x, prepend=0)                        # crude high-pass
    return 0.18 * x * _env(n, 0.012)


def tone(freq, dur, amp, sr=SR, decay=None):
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = amp * (np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t))
    a = np.minimum(1, t / 0.01)                      # 10 ms attack, avoids clicks
    return x * a * (_env(n, decay) if decay else 1)


def make(bpm: float, first: float, bars: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    total = first + bars * 4 * beat + 2.0
    y = np.zeros(int(total * SR) + SR)

    def add(x, t):
        t += rng.uniform(-0.003, 0.003)              # human jitter
        i = int(round(t * SR))
        if i < 0:
            return
        y[i:i + len(x)] += x[: max(0, len(y) - i)]

    roots = [55.0, 43.65, 49.0, 41.2]                 # A1 F1 G1 E1 (one per bar)
    chords = [[220, 261.6, 329.6], [174.6, 220, 261.6], [196, 246.9, 293.7], [164.8, 207.7, 246.9]]
    k, s = kick(), snare(rng)
    breakdown = range(bars // 2, bars // 2 + 4)       # 4 bars, no drums

    # intro pad starting before bar 1, plus a snare pickup on the last 8th
    for c in chords[3]:
        add(tone(c, max(first, 0.05), 0.04), 0)
    if first - beat / 2 > 0:
        add(s * 0.6, first - beat / 2)

    for bar in range(bars):
        t0 = first + bar * 4 * beat
        for c in chords[bar % 4]:
            add(tone(c, 4 * beat, 0.035), t0)        # sustained pad: changes on downbeats
        if bar in breakdown:
            continue
        for b in range(4):
            tb = t0 + b * beat
            add(tone(roots[bar % 4], beat * 0.9, 0.22, decay=0.25), tb)
            if b in (0, 2):
                add(k, tb)
            if b in (1, 3):
                add(s, tb)
            add(hat(rng), tb + beat / 2)
            if b == 0:
                add(hat(rng) * 2.5, tb)               # crash-ish accent on the downbeat
        if bar % 2 == 1:
            add(k * 0.7, t0 + 2.5 * beat)             # syncopated kick

    y /= np.abs(y).max() * 1.1
    return y


def write_mp3(y: np.ndarray, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "x.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
            w.writeframes((y * 32767).astype("<i2").tobytes())
        subprocess.run([config.FFMPEG, "-loglevel", "error", "-y", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-q:a", "2", str(out)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bpm", type=float, default=150)
    ap.add_argument("--first", type=float, default=1.0)
    ap.add_argument("--bars", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    write_mp3(make(a.bpm, a.first, a.bars, a.seed), Path(a.out))
    print(f"wrote {a.out}: bpm={a.bpm} first={a.first} bars={a.bars}")


if __name__ == "__main__":
    main()