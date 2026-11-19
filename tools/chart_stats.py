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

from backend.simai.reader import read_maidata
from backend.simai.stats import stats

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