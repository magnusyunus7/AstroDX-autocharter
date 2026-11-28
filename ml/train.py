"""Train the chart model (Phase C3), resumable.

    python -m ml.train --data ~/dataset --out ~/runs/mid_both --size mid --features both

Run the same command again to continue after a stop (from the last save,
every --save-every steps). Each evaluation (every --eval-every steps) scores
the validation charts and writes <out>/log.jsonl:
  loss       next-token loss on all validation charts (lower = better)
  loss_13    the same on validation charts of level 13 and up
  acc        share of next tokens guessed right
The best model so far (lowest loss) is <out>/best.pt. Training stops by
itself when the loss hasn't improved for --patience evaluations, or at --steps.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                       # noqa: E402
import torch                                                             # noqa: E402
import torch.nn.functional as F                                          # noqa: E402

from ml.data import PAD, SongStore, collate, level_weight, load_rows, make_example, to_torch  # noqa: E402
from ml.model import ChartModel                                          # noqa: E402


def keep(batch):
    return batch


class TrainStream(torch.utils.data.IterableDataset):
    """Endless random training examples (charts picked by level weight, rotated / mirrored)."""

    def __init__(self, data: Path, rows: list[dict], features: str, batch: int, seed: int, hide: float = 0.0):
        self.data, self.rows, self.features, self.batch, self.seed = data, rows, features, batch, seed
        self.hide = hide
        self.weights = [level_weight(r["level_value"]) for r in rows]

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1000 + wid)
        store = SongStore(self.data, cache=16)
        while True:
            picks = rng.choices(self.rows, weights=self.weights, k=self.batch)
            yield collate([make_example(r, store, self.features, rng, augment=True, hide=self.hide) for r in picks])


@torch.no_grad()
def evaluate(model, val_batches, device, amp) -> dict:
    model.eval()
    tot = {"all": [0.0, 0, 0], "13": [0.0, 0, 0]}                     # loss sum, tokens, right
    for b in val_batches:
        b = to_torch(b, device)
        with amp:
            logits = model(b)
        tgt = b["tok_out"]
        loss = F.cross_entropy(logits.float().transpose(1, 2), tgt, ignore_index=PAD, reduction="none")
        right = (logits.argmax(-1) == tgt) & (tgt != PAD)
        n = (tgt != PAD)
        for i in range(tgt.shape[0]):
            keys = ["all"] + (["13"] if b["level"][i] >= 13 else [])
            for k in keys:
                tot[k][0] += float(loss[i][n[i]].sum()); tot[k][1] += int(n[i].sum()); tot[k][2] += int(right[i].sum())
    model.train()
    return {"loss": tot["all"][0] / max(1, tot["all"][1]), "acc": tot["all"][2] / max(1, tot["all"][1]),
            "loss_13": tot["13"][0] / max(1, tot["13"][1])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", default="mid", choices=("small", "mid", "large"))
    ap.add_argument("--features", default="both", choices=("mel", "mert", "both"))
    ap.add_argument("--batch", type=int, default=16, help="charts per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dropout", type=float, default=0.1, help="more = less memorising (try 0.2)")
    ap.add_argument("--weight-decay", type=float, default=0.05, help="more = less memorising (try 0.1)")
    ap.add_argument("--hide-bars", type=float, default=0.0,
                    help="share of past bars hidden from the writer while training (makes it listen more; try 0.25)")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--warmup", type=int, default=1500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data, out = args.data.expanduser(), args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cpu", enabled=False)

    lock = out / ".lock"
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
            sys.exit(f"another training run is going in {out}; stop it first")
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    lock.write_text(str(os.getpid()))
    try:
        run(args, data, out, device, amp)
    finally:
        lock.unlink(missing_ok=True)


def run(args, data, out, device, amp) -> None:
    train_rows, val_rows = load_rows(data, "train"), load_rows(data, "val")
    print(f"{len(train_rows)} training charts, {len(val_rows)} validation charts; device {device}", flush=True)
    model = ChartModel(args.size, args.features, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    step, best, bad = 0, float("inf"), 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        if ck["config"] != model.config:
            sys.exit(f"{out} holds a {ck['config']} run; use another --out for {model.config}")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best, bad = ck["step"], ck["best"], ck.get("bad", 0)
        print(f"resumed at step {step} (best validation loss {best:.4f})", flush=True)
    print(f"model: {args.size}, features {args.features}, {model.n_params() / 1e6:.1f}M parameters", flush=True)
    (out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    if bad >= args.patience:
        print("this run already stopped (no improvement); nothing to do", flush=True)
        return

    # validation: every val chart once, from its start, not rotated (same every time)
    vstore = SongStore(data, cache=8)
    val_rows = sorted(val_rows, key=lambda r: r["token_count"])

    def val_batches():
        for i in range(0, len(val_rows), args.batch):
            yield collate([make_example(r, vstore, args.features, None, augment=False)
                           for r in val_rows[i:i + args.batch]])

    stream = TrainStream(data, train_rows, args.features, args.batch, args.seed + step, args.hide_bars)
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=args.workers,
                                         collate_fn=keep,             # keep numpy: sent by pipe, not shared memory
                                         persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))))

    def save(path, extra=None):
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "bad": bad,
                    "config": model.config, **(extra or {})}, tmp)
        tmp.replace(path)

    model.train()
    t0, run_loss, n = time.time(), 0.0, 0
    for b in loader:
        if step >= args.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = to_torch(b, device)
        with amp:
            logits = model(b)
        loss = F.cross_entropy(logits.float().transpose(1, 2), b["tok_out"], ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run_loss += loss.item(); n += 1
        if step % 100 == 0:
            print(f"step {step}  loss {run_loss / n:.4f}  lr {lr_at(step):.2e}  {(time.time() - t0) / n:.2f}s/step",
                  flush=True)
            t0, run_loss, n = time.time(), 0.0, 0
        if step % args.save_every == 0:
            save(out / "last.pt")
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, val_batches(), device, amp)
            msg = f"  validation: loss {ev['loss']:.4f} (level 13+: {ev['loss_13']:.4f})  acc {ev['acc']:.1%}"
            if ev["loss"] < best:
                best, bad = ev["loss"], 0
                save(out / "best.pt", {"eval": ev})
                msg += "  (best so far, saved)"
            else:
                bad += 1
                msg += f"  (no better for {bad} evaluation{'s' if bad > 1 else ''})"
            print(msg, flush=True)
            with open(out / "log.jsonl", "a") as fh:
                fh.write(json.dumps({"step": step, **ev}) + "\n")
            save(out / "last.pt")
            if bad >= args.patience:
                print(f"stopping: no improvement for {bad} evaluations", flush=True)
                break
    save(out / "last.pt")
    print(f"done at step {step}; best validation loss {best:.4f} -> {out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
