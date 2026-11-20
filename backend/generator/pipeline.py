"""Analysis + level -> chart.

budgets (planner) -> grid moments (placement) -> note kinds: tap / hold /
slide / touch + BREAKs (kinds) -> buttons and slide shapes (hands) -> Chart ->
maidata.txt.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np

from backend.generator import hands, kinds, placement, planner
from backend.generator.difficulty import level_value, targets as level_targets
from backend.models.chart import Chart, Hold, Slide, Tap, Touch
from backend.simai.maidata import DifficultyChart, MaidataMeta, build_maidata, rebase_first
from backend.simai.stats import stats

DESIGNER = "AstroDX autocharter"
MAX_SCALE = 1.8            # most extra density planned to make up for notes dropped around holds/slides


def generate(analysis: dict, *, level: str, slot: int = 5, seed: int = 0,
             overrides: dict[str, str] | None = None, title: str = "", artist: str = "",
             duration: float | None = None) -> dict:
    overrides = overrides or {}
    t = level_targets(level)
    bpm, first, k_start = analysis["bpm"], analysis["first"], analysis["k_start"]
    beat = 60.0 / bpm
    sub = placement.SUB

    each_frac = t.each_pct / 100.0
    capacity = placement.bar_capacity(analysis, t, each_frac)

    # chart beat 0 = &first after moving it back to the start of the audio
    first0, bars_added = rebase_first(first, bpm)
    def chart_beat(grid: int) -> Fraction:
        return Fraction(k_start * sub + grid, sub) + 4 * bars_added   # beats from first0

    # BREAK candidates: the first note of each high/peak section, on its downbeat
    levels = planner.section_level_per_bar(analysis, overrides)
    starts = {s["start_bar"] for s in analysis["sections"]
              if levels[s["start_bar"]] in ("high", "peak")} if analysis["sections"] else set()
    break_grids = {b * 4 * sub for b in starts}

    gap = placement.min_gap_steps(analysis, t)
    audio_end = duration if duration else first + (k_start + analysis["n_beats"]) * beat
    end_limit = audio_end - 0.05

    def build(scale: float):
        # holds/slides drop a few notes to free a hand: plan a little more to make up for it
        plan_t = replace(t, nps=t.nps * scale, peak_nps=t.peak_nps * scale)
        budgets = planner.note_budgets(analysis, plan_t, overrides, capacity)
        ms = [m for m in placement.place(analysis, t, budgets, mark_each=False)
              if chart_beat(m.grid) >= 0]
        tms = [first0 + float(chart_beat(m.grid)) * beat for m in ms]
        return kinds.decide(ms, tms, analysis, t, gap, seed=seed, break_grids=break_grids,
                            end_limit=end_limit)

    def hits(ms):
        return sum(2 if m.each else 1 for m in ms)

    moments, times, ks = build(1.0)
    planned = t.nps * (times[-1] - times[0]) if len(times) > 1 else 0
    scale = 1.0
    for _ in range(2):
        if not hits(moments) or hits(moments) >= 0.97 * planned:
            break
        scale = min(MAX_SCALE, scale * planned / hits(moments))
        moments, times, ks = build(scale)
    strong_at = {tm: m.strength >= 1.0 and m.grid % sub == 0 for m, tm in zip(moments, times)}
    placed = hands.assign(times, [m.each for m in moments], seed=seed, kinds=ks, beat=beat,
                          allow_wifi=level_value(level) >= 13.7, strong=lambda tm: strong_at.get(tm, False))

    chart = Chart(bpm)
    notes_out = []
    for m, k, group, tm in zip(moments, ks, placed, times):
        b0 = chart_beat(m.grid)
        brk = k.brk
        kinds_here, end_t = [], tm
        for p in group:
            if p.kind == "hold":
                chart.add(Hold(b0, p.button, k.length, brk=brk))
                end_t = max(end_t, tm + float(k.length) * beat)
            elif p.kind == "slide":
                chart.add(Slide(b0, p.button, p.shape, p.end, k.length, brk=brk))
                end_t = max(end_t, tm + float(kinds.SLIDE_WAIT + k.length) * beat)
            elif p.kind == "touch":
                chart.add(Touch(b0, p.sensor))
            else:
                chart.add(Tap(b0, p.button, brk=brk))
            kinds_here.append(p.kind)
        notes_out.append({"t": round(tm, 4), "buttons": [p.button or p.sensor for p in group],
                          "kinds": kinds_here, "end": round(end_t, 4),
                          "break": brk and any(x != "touch" for x in kinds_here)})

    demo = analysis.get("demo") or {}
    meta = MaidataMeta(title=title or "untitled", artist=artist, first=first0, designer=DESIGNER,
                       extra={"demo_seek": f"{demo.get('seek', 0)}", "demo_len": f"{demo.get('len', 15)}"}
                       if demo else {})
    maidata = build_maidata(meta, [DifficultyChart(slot=slot, level=level, chart=chart)],
                            audio_duration=duration)
    body = maidata.split(f"&inote_{slot}=", 1)[1]
    got = stats(body, first0)
    return {
        "maidata": maidata,
        "notes": notes_out,
        "targets": t.to_dict(),
        "stats": {k: round(v, 2) if isinstance(v, float) else v for k, v in got.items()},
        "level": level, "slot": slot, "seed": seed,
    }