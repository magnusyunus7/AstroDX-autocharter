"""Analysis + level -> chart (v1: taps, EACH, breaks).

budgets (planner) -> grid moments (placement) -> buttons (hands) -> Chart ->
maidata.txt. Holds, slides and touch notes come in later steps.
"""
from __future__ import annotations

from fractions import Fraction

from backend.generator import hands, placement, planner
from backend.generator.difficulty import targets as level_targets
from backend.models.chart import Chart, Tap
from backend.simai.maidata import DifficultyChart, MaidataMeta, build_maidata, rebase_first
from backend.simai.stats import stats

DESIGNER = "AstroDX autocharter"


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
    budgets = planner.note_budgets(analysis, t, overrides, capacity)
    moments = placement.place(analysis, t, budgets)

    # chart beat 0 = &first after moving it back to the start of the audio
    first0, bars_added = rebase_first(first, bpm)
    def chart_beat(grid: int) -> Fraction:
        return Fraction(k_start * sub + grid, sub) + 4 * bars_added   # beats from first0

    moments = [m for m in moments if chart_beat(m.grid) >= 0]
    times = [first0 + float(chart_beat(m.grid)) * beat for m in moments]
    buttons = hands.assign(times, [m.each for m in moments], seed=seed)

    # BREAK on the first note of each high/peak section, if it lands on its downbeat
    levels = planner.section_level_per_bar(analysis, overrides)
    starts = {s["start_bar"] for s in analysis["sections"]
              if levels[s["start_bar"]] in ("high", "peak")} if analysis["sections"] else set()
    break_grids = {b * 4 * sub for b in starts}

    chart = Chart(bpm)
    notes_out = []
    for m, btns, tm in zip(moments, buttons, times):
        brk = m.grid in break_grids
        for btn in btns:
            chart.add(Tap(chart_beat(m.grid), btn, brk=brk))
        notes_out.append({"t": round(tm, 4), "buttons": btns, "break": brk})

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