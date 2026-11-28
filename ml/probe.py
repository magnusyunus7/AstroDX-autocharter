"""How much does the model listen to the music? (Phase C3 diagnosis)

    python -m ml.probe --data ~/dataset --model ~/runs/mid_reg/best.pt

Scores the validation charts three times with the same model:
  real audio      the normal score (as in training's validation)
  no audio        the song's features replaced with silence
  wrong song      every chart paired with another validation song's audio
If the score barely gets worse without the right audio, the model writes
mostly from its own previous tokens (and will loop); if it gets much worse,
it does follow the music.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.nn.functional as F                                          # noqa: E402

from ml.data import MEL_DIM, PAD, SongStore, collate, load_rows, make_example, to_torch   # noqa: E402
from ml.generate import load_model                                       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    data = args.data.expanduser()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.model.expanduser(), device)
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cpu", enabled=False)
    rows = sorted(load_rows(data, "val"), key=lambda r: r["token_count"])
    store = SongStore(data, cache=8)
    print(f"{len(rows)} validation charts; model from step {ck['step']}", flush=True)
    examples = [make_example(r, store, model.features, None, augment=False) for r in rows]
    silence = np.zeros(MEL_DIM + 256, np.float16)
    silence[:MEL_DIM] = -1.0                                           # the mel value for silence
    rng = np.random.default_rng(0)
    other = rng.permutation(len(examples))
    while any(other[i] == i for i in range(len(other))):
        other = rng.permutation(len(examples))

    def variant(i, how):
        e = dict(examples[i])
        if how == "no audio":
            e["enc"] = np.broadcast_to(silence, e["enc"].shape).copy()
        elif how == "wrong song":
            src = examples[other[i]]["enc"]
            n = len(e["enc"])
            reps = int(np.ceil(n / len(src)))
            e["enc"] = np.concatenate([src] * reps)[:n]
        return e

    for how in ("real audio", "no audio", "wrong song"):
        tot = {"all": [0.0, 0], "13": [0.0, 0]}
        with torch.no_grad():
            for i in range(0, len(examples), args.batch):
                idx = list(range(i, min(i + args.batch, len(examples))))
                b = to_torch(collate([variant(j, how) for j in idx]), device)
                with amp:
                    logits = model(b)
                tgt = b["tok_out"]
                loss = F.cross_entropy(logits.float().transpose(1, 2), tgt, ignore_index=PAD, reduction="none")
                for k, j in enumerate(idx):
                    m = tgt[k] != PAD
                    l = float(loss[k][m].sum()); n = int(m.sum())
                    tot["all"][0] += l; tot["all"][1] += n
                    if rows[j]["level_value"] >= 13:
                        tot["13"][0] += l; tot["13"][1] += n
        a, t13 = tot["all"][0] / tot["all"][1], tot["13"][0] / max(1, tot["13"][1])
        if how == "real audio":
            base = (a, t13)
            print(f"  {how:12}: loss {a:.4f} (13+: {t13:.4f})", flush=True)
        else:
            print(f"  {how:12}: loss {a:.4f} (13+: {t13:.4f})  ->  {a - base[0]:+.4f} "
                  f"({(a / base[0] - 1) * 100:+.0f}%) worse without the right audio", flush=True)


if __name__ == "__main__":
    main()
