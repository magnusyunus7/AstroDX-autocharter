// Phase 2 analysis view: grid onsets per stem (drums / bass / melody) drawn as
// lanes at the bottom of the waveform, and bar-by-bar grid fit drawn as a
// strip under the ruler (red = hits don't sit on the grid, grey = too quiet
// to judge, yellow = loose/irregular rhythm like rolls or glitches). Analysis
// runs on the *confirmed* timing.

import { api, fmtTime } from "./api.js";

const $ = (id) => document.getElementById(id);
// min: grid values (0-255) below this aren't drawn (melody's chroma curve is noisier)
const LANES = [
  { stem: "drums", color: "#e3e6ee", label: "drums", min: 25 },
  { stem: "bass", color: "#ffab40", label: "bass", min: 25 },
  { stem: "melody", color: "#69f0ae", label: "melody", min: 50 },
];
const LANE_H = 13;

export class AnalysisView {
  constructor(timeline, timing) {
    this.tl = timeline;
    this.timing = timing;
    this.data = null;
    this.projectId = null;
    this.showOnsets = true;
    timeline.overlays.push((g, v) => this._draw(g, v));
    timing.onChange = () => this._syncWithTiming();
    $("show-onsets").onchange = (e) => { this.showOnsets = e.target.checked; this.tl.draw(); };
    $("btn-reanalyze").onclick = () => this.load(this.timing.project, { force: true });
  }

  /** Analysis only describes the *confirmed* timing. While the user has
   *  unconfirmed edits, don't offer stale results or a re-run on the old grid. */
  _syncWithTiming() {
    const btn = $("btn-reanalyze");
    if (!this.timing.project?.timing?.confirmed) return;   // load() handles this case
    if (this.loading) return;
    if (this.timing.dirty) {
      btn.disabled = true;
      this._info("Timing changed since the analysis: press Confirm timing to re-analyze.", "warn-text");
    } else {
      btn.disabled = false;
      if (this.data) this._summarize();
    }
  }

  async load(project, { force = false } = {}) {
    this.projectId = project.id;
    this.data = null;
    this.tl.draw();
    const btn = $("btn-reanalyze");
    if (!project.timing?.confirmed) {
      this._info("Analysis runs once the timing is confirmed.", "");
      btn.hidden = true;
      return;
    }
    btn.hidden = false;
    btn.disabled = true;
    this.loading = true;
    this._info("Analyzing onsets and grid fit… (about 10–20 s the first time)", "");
    try {
      const d = await api.analyze(project.id, force);
      if (this.projectId !== project.id) return;
      this.data = d;
      this._summarize();
    } catch (e) {
      this._info("Analysis failed: " + e.message, "warn-text");
    } finally {
      this.loading = false;
      this._syncWithTiming();
      this.tl.draw();
    }
  }

  /** Bar number as shown on the grid (bar tags) for analysis bar index b. */
  barNo(b) { return this.data.k_start / 4 + b + 1; }

  _summarize() {
    const d = this.data, s = d.summary;
    const n = d.bars.length;
    const pl = (k, word) => `${k} ${word}${k === 1 ? "" : "s"}`;
    const irr = s.irregular_bars ? ` ${pl(s.irregular_bars, "bar")} with loose rhythm (yellow: rolls, ` +
      `glitches, free-time parts) can't be judged.` : "";
    if (s.drift) {
      this._info(`⚠ From bar ${this.barNo(s.from_bar)} (${fmtTime(s.from_time).slice(0, 5)}) the hits stop ` +
        `matching the grid: the tempo changes there, or the BPM is slightly off. ` +
        `${s.off_bars} of ${n} bars affected (red).` + irr, "bad-text");
    } else if (s.off_bars > 0) {
      this._info(`Analysis: the grid fits the song; ${pl(s.off_bars, "isolated bar")} marked red ` +
        `(worth a quick listen).` + irr, "warn-text");
    } else {
      this._info(`✓ Analysis: the grid fits the song (${n} bars).` + irr, "ok-text");
    }
  }

  _info(text, cls) {
    const el = $("analysis-info");
    el.textContent = text;
    el.className = "small " + (cls || "hint");
  }

  _draw(g, { w, h, rulerH, tl }) {
    const d = this.data;
    if (!d) return;
    const stale = this.timing.dirty;          // grid edited since this analysis
    const beat = 60 / d.bpm;
    const tAt = (k) => d.first + k * beat;    // k = beat index relative to `first`

    // bar fit strip under the ruler (+ faint tint over "off" bars)
    if (!stale) {
      for (const bar of d.bars) {
        if (bar.status === "ok") continue;
        const x0 = tl.timeToX(tAt(d.k_start + 4 * bar.bar));
        const x1 = tl.timeToX(tAt(d.k_start + 4 * bar.bar + 4));
        if (x1 < 0 || x0 > w) continue;
        if (bar.status === "off") {
          g.fillStyle = "#ff52521c"; g.fillRect(x0, rulerH, x1 - x0, h - rulerH);
          g.fillStyle = "#ff5252"; g.fillRect(x0, rulerH, x1 - x0, 4);
        } else if (bar.status === "irregular") {
          g.fillStyle = "#ffd54f"; g.fillRect(x0, rulerH, x1 - x0, 4);
        } else {
          g.fillStyle = "#ffffff30"; g.fillRect(x0, rulerH, x1 - x0, 4);
        }
      }
    }

    if (!this.showOnsets) return;
    // onset lanes at the bottom
    const sub = d.subdiv;
    const stepPx = (beat / sub) * tl.pxPerSec;
    const iFrom = Math.max(0, Math.floor(((tl.start - d.first) / beat - d.k_start) * sub) - 1);
    const iTo = Math.min(d.n_beats * sub, Math.ceil(((tl.start + tl.viewSec - d.first) / beat - d.k_start) * sub) + 1);
    const top = h - LANES.length * LANE_H - 2;
    g.fillStyle = "#0d0e11cc";
    g.fillRect(0, top - 2, w, LANES.length * LANE_H + 4);
    g.font = "10px ui-monospace, Consolas, monospace";
    g.textBaseline = "middle";
    LANES.forEach((lane, li) => {
      const vals = d.stems[lane.stem];
      const y0 = top + li * LANE_H;
      g.fillStyle = lane.color;
      let lastX = -1, best = 0;
      const flush = (x) => {
        if (best >= lane.min) {
          const hh = Math.max(2, (best / 255) * (LANE_H - 2) * 1.6);
          g.globalAlpha = stale ? 0.3 : 0.35 + 0.65 * Math.min(1, best / 160);
          g.fillRect(x, y0 + LANE_H - 1 - Math.min(hh, LANE_H - 2), Math.max(1, Math.min(3, stepPx - 1)), Math.min(hh, LANE_H - 2));
        }
      };
      for (let i = iFrom; i < iTo; i++) {
        const x = Math.round(tl.timeToX(tAt(d.k_start + i / sub)));
        if (x !== lastX) { if (lastX >= 0) flush(lastX); lastX = x; best = 0; }
        if (vals[i] > best) best = vals[i];
      }
      if (lastX >= 0) flush(lastX);
      g.globalAlpha = 1;
      g.fillStyle = lane.color;
      g.fillText(lane.label, 4, y0 + LANE_H / 2);
    });
    if (stale) {
      g.fillStyle = "#ffb74d"; g.textBaseline = "top";
      g.fillText("timing changed: confirm to re-analyze", 70, top - 14);
    }
  }
}