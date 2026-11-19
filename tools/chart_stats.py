"""Chart statistics: what a level looks like in numbers.

    python -m tools.chart_stats samples/

For every chart (every difficulty in every maidata.txt): level, notes per
second (average and busiest 4 s), note-type mix, EACH share, longest stream,
slides per minute. Then averages per level, which become the generator's
density targets.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

from backend.simai.reader import note_groups, read_maidata

SLIDE_CHARS = set("-<>^vpqszwV")


def classify(tok: str) -> list[str]:
    """Note kinds in one token (a slide token with '*' has several paths)."""
    t = re.sub(r"\[[^\]]*\]", "", tok)          # drop durations
    kinds = []
    brk = "b" in t
    if t[:1] in "ABCDE":
        kinds.append("touch_hold" if "h" in t else "touch")
    elif any(c in SLIDE_CHARS for c in t[1:]):
        kinds.append("slide_star")              # the star you tap
        kinds += ["slide"] * (t.count("*") + 1)
    elif "h" in t:
        kinds.append("hold")
    else:
        kinds.append("tap")
    if brk:
        kinds.append("break")
    if "x" in t:
        kinds.append("ex")
    return kinds


def stats(body: str, first: float) -> dict:
    groups = note_groups(body, first)
    if not groups:
        return {}
    times = np.array([t for t, _ in groups])
    sizes = np.array([len(n) for _, n in groups])
    kinds: dict[str, int] = {}
    for _, notes in groups:
        for tok in notes:
            for k in classify(tok):
                kinds[k] = kinds.get(k, 0) + 1
    hits = sizes.sum()                           # notes you hit (slide paths not counted twice)
    span = max(1.0, times[-1] - times[0])
    # busiest 4 s window
    peak = 0
    j = 0
    for i in range(len(times)):
        while times[i] - times[j] > 4.0:
            j += 1
        peak = max(peak, sizes[j:i + 1].sum())
    # longest stream: consecutive groups <= 1/8 s apart (fast runs)
    gaps = np.diff(times)
    run = best = 1
    for g in gaps:
        run = run + 1 if g <= 0.125 else 1
        best = max(best, run)
    return {
        "notes": int(hits),
        "nps": hits / span,
        "peak_nps": peak / 4.0,
        "each_pct": 100 * (sizes > 1).sum() / len(sizes),
        "tap_pct": 100 * kinds.get("tap", 0) / hits,
        "hold_pct": 100 * kinds.get("hold", 0) / hits,
        "slide_pm": kinds.get("slide", 0) / span * 60,
        "touch_pct": 100 * (kinds.get("touch", 0) + kinds.get("touch_hold", 0)) / hits,
        "break_pct": 100 * kinds.get("break", 0) / hits,
        "longest_stream": int(best),
        "length_s": span,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--fit", action="store_true",
                    help="print new FIT constants for backend/generator/difficulty.py")
    a = ap.parse_args()
    rows = []
    for md_path in sorted(Path(a.folder).rglob("maidata.txt")):
        text = md_path.read_text(encoding="utf-8", errors="replace")
        md = read_maidata(text)
        for slot, body in sorted(md["charts"].items()):
            lv = re.search(rf"^&lv_{slot}=(.*)$", text, re.M)
            level = lv.group(1).strip() if lv else "?"
            s = stats(body, md["first"])
            if s:
                rows.append((md_path.parent.name, slot, level, s))

    cols = ("notes", "nps", "peak_nps", "each_pct", "tap_pct", "hold_pct", "slide_pm",
            "touch_pct", "break_pct", "longest_stream")
    head = f"{'chart':20} {'slot':>4} {'lv':>4} " + " ".join(f"{c:>9}" for c in cols)
    print(head)
    for name, slot, level, s in rows:
        print(f"{name[:20]:20} {slot:>4} {level:>4} " + " ".join(f"{s[c]:9.1f}" for c in cols))

    print("\nper level (mean):")
    by: dict[str, list[dict]] = {}
    for _, _, level, s in rows:
        by.setdefault(level, []).append(s)
    def key(lv):
        m = re.match(r"(\d+)(\+?)", lv)
        return (int(m.group(1)), m.group(2)) if m else (99, "")
    print(f"{'lv':>5} {'n':>3} " + " ".join(f"{c:>9}" for c in cols))
    for lv in sorted(by, key=key):
        ss = by[lv]
        print(f"{lv:>5} {len(ss):>3} " + " ".join(f"{np.mean([s[c] for s in ss]):9.1f}" for c in cols))
    if a.fit:
        print_fit(rows)


def print_fit(rows, min_level: float = 10.0):
    """Robust straight-line fits (outliers trimmed) of each stat vs level."""
    from backend.generator.difficulty import level_value
    L = np.array([level_value(r[2]) if r[2] != "?" else np.nan for r in rows])
    keys = (("nps", True), ("peak_nps", True), ("slide_pm", False), ("each_pct", False),
            ("hold_pct", False), ("touch_pct", False), ("break_pct", False), ("longest_stream", False))
    print("\nFIT = {")
    for key, log in keys:
        y = np.array([r[3][key] for r in rows], float)
        m = np.isfinite(L) & (L >= min_level) & ((y > 0) | (not log))
        x, yy = L[m], (np.log(y[m]) if log else y[m])
        keep = np.ones(len(x), bool)
        for _ in range(3):
            b, a0 = np.polyfit(x[keep], yy[keep], 1)
            r = yy - (a0 + b * x)
            keep = np.abs(r) <= 1.5 * np.std(r[keep]) + 1e-9
        print(f'    "{key}": ({a0:.4f}, {b:.4f}, {log}),   # {keep.sum()}/{len(x)} charts used')
    print("}")


if __name__ == "__main__":
    main()