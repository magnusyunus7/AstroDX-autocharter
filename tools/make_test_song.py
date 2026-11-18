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


def make(bpm: float, first: float, bars: int, seed: int = 1, events: list | None = None,
         triplet_bars: range = range(0), glitch_bars: range = range(0)) -> np.ndarray:
    """Render the song. If `events` is a list, (time, kind) of every sound is
    appended to it (kinds: kick, snare, hat, bass, chord, pad).
    triplet_bars: bars played as a shuffle (kick/snare/hats on triplet positions).
    glitch_bars: bars of irregular stutter hits at random times (no grid)."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    total = first + bars * 4 * beat + 2.0
    y = np.zeros(int(total * SR) + SR)

    def add(x, t, kind=""):
        t += rng.uniform(-0.003, 0.003)              # human jitter
        if events is not None and kind:
            events.append((t, kind))
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
            add(tone(c, 4 * beat, 0.035), t0, "chord")  # sustained pad: changes on downbeats
        if bar in breakdown:
            continue
        if bar in glitch_bars:
            for _ in range(14):                       # random stutters, loud, off any grid
                add((k if rng.random() < 0.5 else s) * rng.uniform(0.6, 1.0),
                    t0 + rng.uniform(0, 4 * beat), "glitch")
            continue
        if bar in triplet_bars:
            for b in range(4):
                tb = t0 + b * beat
                add(tone(roots[bar % 4], beat * 0.9, 0.22, decay=0.25), tb, "bass")
                add(k if b % 2 == 0 else s, tb, "kick" if b % 2 == 0 else "snare")
                add(hat(rng) * 1.5, tb + beat / 3, "hat")
                add(k * 0.8, tb + 2 * beat / 3, "kick")  # shuffle kick
            continue
        for b in range(4):
            tb = t0 + b * beat
            add(tone(roots[bar % 4], beat * 0.9, 0.22, decay=0.25), tb, "bass")
            if b in (0, 2):
                add(k, tb, "kick")
            if b in (1, 3):
                add(s, tb, "snare")
            add(hat(rng), tb + beat / 2, "hat")
            if b == 0:
                add(hat(rng) * 2.5, tb)               # crash-ish accent on the downbeat
        if bar % 2 == 1:
            add(k * 0.7, t0 + 2.5 * beat, "kick")     # syncopated kick

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


# ---------------------------------------------------------------- structured songs
# Each style differs in instruments, chords and loudness, like real sections.
PROG = {
    "P1": [[220, 261.6, 329.6], [174.6, 220, 261.6], [196, 246.9, 293.7], [164.8, 207.7, 246.9]],
    "P2": [[293.7, 370, 440], [246.9, 293.7, 370], [329.6, 415.3, 493.9], [220, 277.2, 329.6]],
}
ROOTS = {"P1": [55.0, 43.65, 49.0, 41.2], "P2": [73.4, 61.7, 82.4, 55.0]}
STYLES = {
    #          drums   hats   bass   lead   prog   pad_amp
    "intro":  (None,   "4th", False, False, "P1", 0.05),
    "verse":  ("half", "8th", True,  False, "P1", 0.035),
    "chorus": ("full", "16th", True, True,  "P2", 0.045),
    "break":  (None,   None,  False, False, "P1", 0.06),
}


def riser(dur, sr=SR, rng=None):
    """Noise sweep getting louder and brighter: the classic build into a drop."""
    n = int(dur * sr)
    rng = rng or np.random.default_rng(0)
    x = rng.standard_normal(n)
    env = np.linspace(0, 1, n) ** 2
    hp = np.diff(x, prepend=0) * np.linspace(0.3, 1, n)       # brighter towards the end
    return 0.25 * (0.5 * x + hp) * env


def make_structured(bpm: float, first: float, sections: list[tuple[str, int]], seed: int = 1,
                    events: list | None = None, transitions: bool = False,
                    variation: bool = False) -> tuple[np.ndarray, list[int]]:
    """Render a song from a section plan, e.g. [("intro", 8), ("verse", 8), ("chorus", 8)].
    Returns (audio, boundary bar indices). Bar 0 starts at `first`.
    transitions: like real songs, the last bar before each chorus is a fill
    (16th snare roll + riser) and each section starts with a crash + kick impact.
    variation: like real songs, sections aren't uniform inside: every 4 bars
    the lead/hats/chord voicing change a little (must NOT become new sections)."""
    rng = np.random.default_rng(seed)
    beat = 60 / bpm
    bars = sum(n for _, n in sections)
    y = np.zeros(int((first + bars * 4 * beat + 2.0) * SR) + SR)

    def add(x, t, kind=""):
        t += rng.uniform(-0.003, 0.003)
        if events is not None and kind:
            events.append((t, kind))
        i = int(round(t * SR))
        if 0 <= i < len(y):
            y[i:i + len(x)] += x[: len(y) - i]

    k, s = kick(), snare(rng)
    boundaries, bar = [], 0
    for si, (style, n) in enumerate(sections):
        boundaries.append(bar)
        drums, hats, bass, lead, prog, pad = STYLES[style]
        next_style = sections[si + 1][0] if si + 1 < len(sections) else None
        for i in range(n):
            t0 = first + bar * 4 * beat
            sub_phrase = (i // 4) % 2 if variation else 0     # small changes every 4 bars
            if transitions and i == 0 and si > 0:
                add(hat(rng) * 4, t0); add(k, t0)                     # impact: crash + kick
            if transitions and i == n - 1 and next_style == "chorus":
                for j in range(16):                                    # fill bar: snare roll + riser
                    add(s * (0.4 + 0.6 * j / 15), t0 + j * beat / 4)
                add(riser(4 * beat, rng=rng), t0)
                bar += 1
                continue
            chord = PROG[prog][bar % 4]
            for c in chord:
                add(tone(c * (2 if sub_phrase else 1), 4 * beat, pad * (0.8 if sub_phrase else 1)), t0, "chord")
            for b in range(4):
                tb = t0 + b * beat
                if bass:
                    add(tone(ROOTS[prog][bar % 4], beat * 0.9, 0.2, decay=0.25), tb, "bass")
                if drums == "full" or (drums == "half" and b in (0, 2)):
                    add(k, tb, "kick")
                if drums == "full" and b in (1, 3) or drums == "half" and b == 2:
                    add(s, tb, "snare")
                steps = {"4th": [0], "8th": [0.5], "16th": [0, 0.25, 0.5, 0.75]}.get(hats, [])
                for st in steps:
                    add(hat(rng), tb + st * beat, "hat")
                if variation and sub_phrase and hats:
                    add(hat(rng) * 0.7, tb + 0.75 * beat, "hat")     # extra offbeat hat
                if lead:
                    for st in ((0, 0.5) if not sub_phrase else (0, 0.25, 0.5, 0.75)):
                        note = chord[(b * 2 + int(st * 2)) % 3] * 2
                        add(tone(note, beat * 0.45, 0.06, decay=0.15), tb + st * beat, "lead")
            bar += 1
    boundaries.append(bar)
    y /= np.abs(y).max() * 1.1
    return y, boundaries