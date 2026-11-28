"""Write charts with a trained chart model (Phase C3/C4).

The model writes tokens one at a time; ml/grammar.py masks anything that
would not decode (or would put two notes on one button, or an impossible
slide), so every chart it finishes is a valid chart. Sampling: temperature
and top-p (nucleus) over what's allowed.

Loop guard (optional): if a bar pattern (or a pair of bars) has just repeated
GUARD_REPEATS times in a row, the button / sensor / slide-shape choices that
would repeat it once more are made less likely. The rhythm is never pushed,
only where the notes go; charts do repeat a pattern for a few bars.

Long charts: the decoder sees up to MAX_TOKENS tokens. When a chart gets
longer, it keeps writing with the last CONTEXT tokens (cut at a BEAT) as
its memory; the encoder still hears the whole song.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import torch

from backend.chartcodec.notes import ChartData
from backend.chartcodec.tokens import INDEX, decode, from_ids
from ml.data import (BEAT, BOS, EOS, MAX_STEPS, MAX_TOKENS, PAD, STEPS_PER_BEAT, beats_to_seconds,
                     bpm_bucket, encoder_features, to_torch)
from ml.grammar import Grammar
from ml.model import ChartModel

CONTEXT = 1024
GUARD_REPEATS = 3                  # a bar (or 2-bar pair) repeated this many times in a row ...
GUARD_TOKENS = ("B", "S", "SH_")   # ... gets these choices discouraged (where notes go, not when)


def _guard_target(bars: dict, cb: int, prefix: list[int]) -> int | None:
    """If the bars just before bar cb repeat one pattern (period 1 or 2 bars)
    GUARD_REPEATS times and bar cb is on its way to repeating it again, the
    token that would continue the repeat (else None)."""
    for q in (1, 2):
        pattern = bars.get(cb - q)
        if not pattern:
            continue
        if all(bars.get(cb - q * k) == pattern for k in range(1, GUARD_REPEATS + 1)):
            if len(prefix) < len(pattern) and pattern[:len(prefix)] == prefix:
                return pattern[len(prefix)]
    return None


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = ChartModel(cfg["size"], cfg["features"], cfg.get("dropout", 0.1)).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, ck


def song_beats(tempo, first: float, duration: float) -> int:
    """How many whole beats of the tempo map fit in the audio after &first."""
    b = np.arange(0, MAX_STEPS // STEPS_PER_BEAT + 1, dtype=np.float64)
    s = first + beats_to_seconds(tempo, b)
    return int(np.searchsorted(s, duration - 0.25))


@torch.no_grad()
def generate(model, device, mel, mert, first: float, tempo: list[tuple[float, float]], duration: float,
             level: float, slot: int = 5, era: str = "DX", source: str = "proofread",
             temperature: float = 0.9, top_p: float = 0.92, seed: int = 0, max_tokens: int = 9000,
             loop_guard: float = 0.0):
    """One chart for one song. Returns (notes, tokens, stats). loop_guard: how
    strongly to discourage a 4th repeat in a row (0 = off; 2.0 = about 7x less likely)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_beats = min(song_beats(tempo, first, duration), MAX_STEPS // STEPS_PER_BEAT - 4)
    n_steps = n_beats * STEPS_PER_BEAT + 16
    enc = encoder_features(mel, mert, first, 0.0, tempo, n_steps, model.features)
    batch = to_torch({"enc": enc[None], "enc_mask": np.zeros((1, n_steps), bool),
                      "bar": (np.arange(n_steps) % (4 * STEPS_PER_BEAT))[None].astype(np.int64)}, device)
    memory = model.encode(batch["enc"], batch["enc_mask"], batch["bar"])
    cond = {k: torch.tensor([v], device=device) for k, v in (
        ("level", float(level)), ("slot", int(slot)), ("era", 0 if era == "STD" else 1),
        ("source", 0 if source == "converted" else 1), ("bpm", bpm_bucket(tempo[0][1])))}
    cond["level"] = cond["level"].float()

    g = Grammar()
    toks, steps = [BOS], [0]
    masked, last_top, guarded = 0, 1, 0
    from backend.chartcodec.tokens import VOCAB
    bars: dict[int, list[int]] = {}
    while not g.done and len(toks) < max_tokens:
        # the window the decoder sees
        if len(toks) > MAX_TOKENS:
            cut = len(toks) - CONTEXT
            while cut < len(toks) and toks[cut] != BEAT:
                cut += 1
            w_tok, w_step = toks[cut:], steps[cut:]
        else:
            w_tok, w_step = toks, steps
        ti = torch.tensor([w_tok], device=device)
        ts = torch.tensor([w_step], device=device).clamp(max=n_steps - 1)
        logits = model.decode(memory, batch["enc_mask"], ti, ts, cond["level"], cond["slot"], cond["era"],
                              cond["source"], cond["bpm"])[0, -1].float().cpu()
        if loop_guard > 0:
            cb = max(0, g.beat_no) // 4
            tgt = _guard_target(bars, cb, bars.get(cb, []))
            if tgt is not None and VOCAB[tgt].startswith(GUARD_TOKENS):
                logits[tgt] -= loop_guard
                guarded += 1
        allow = torch.from_numpy(g.allowed())
        if g.beat_no >= n_beats:                                    # past the song: only finish
            only = torch.zeros_like(allow)
            only[EOS] = True
            if (allow & only).any():
                allow = allow & only
        allow[PAD] = False
        masked += int((logits.argmax() != logits.masked_fill(~allow, -math.inf).argmax()).item())
        logits = logits.masked_fill(~allow, -math.inf) / max(1e-4, temperature)
        probs = torch.softmax(logits, -1)
        if top_p < 1.0:
            sp, si = probs.sort(descending=True)
            keep = sp.cumsum(0) - sp < top_p
            p2 = torch.zeros_like(probs)
            p2[si[keep]] = sp[keep]
            probs = p2 / p2.sum()
        t = int(torch.multinomial(probs, 1, generator=gen))
        g.push(t)
        toks.append(t)
        if t != BEAT:
            bars.setdefault(max(0, g.beat_no) // 4, []).append(t)
        steps.append(int(math.floor((max(0, g.beat_no) + float(g.pos or 0)) * STEPS_PER_BEAT)))
        if g.state == "top":
            last_top = len(toks)
    if not g.done:                                                  # ran out of room: drop the unfinished note
        toks = toks[:last_top] + [EOS]
    notes = decode(from_ids(toks))
    return notes, toks, {"tokens": len(toks), "beats": n_beats, "grammar_corrections": masked,
                         "loop_guard_nudges": guarded}


def chart_text(notes, tempo) -> str:
    from backend.chartcodec.write import write_chart
    return write_chart(ChartData([(Fraction(b).limit_denominator(10 ** 6), v) for b, v in tempo], notes))


def maidata(title: str, artist: str, first: float, bpm: float, charts: dict) -> str:
    """charts: {slot: (level text, note text)}."""
    lines = [f"&title={title}", f"&artist={artist}", f"&wholebpm={bpm:g}", f"&first={first:.4f}"]
    for slot, (lv, body) in sorted(charts.items()):
        lines += [f"&lv_{slot}={lv}", f"&inote_{slot}={body}"]
    return "\n".join(lines) + "\n"


__all__ = ["load_model", "generate", "chart_text", "maidata", "INDEX"]
