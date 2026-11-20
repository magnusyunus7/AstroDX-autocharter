// Generate panel: level / difficulty, Generate / Reroll / Export, the
// generated notes drawn as a lane in the music bar, and hit sounds on
// playback (scheduled on the audio clock, like the metronome).

import { api } from "./api.js";

const $ = (id) => document.getElementById(id);
const LOOKAHEAD = 0.12;
const LANE_H = 16;

export class GeneratePanel {
  constructor(player, timeline, timing, analysis) {
    this.player = player;
    this.tl = timeline;
    this.timing = timing;
    this.analysis = analysis;
    this.project = null;
    this.result = null;
    this.seed = null;
    this.showNotes = true;
    this.hitSounds = false;
    this.hitVolume = 0.5;
    this._epoch = -1;
    this._lastCt = 0;
    this._pending = [];
    timeline.overlays.push((g, v) => this._draw(g, v));
    setInterval(() => this._tick(), 25);

    $("g-generate").onclick = () => this.generate();
    $("g-reroll").onclick = () => this.generate({ reroll: true });
    $("show-notes").onchange = (e) => { this.showNotes = e.target.checked; this.tl.draw(); };
    $("hit-sounds").onchange = (e) => { this.hitSounds = e.target.checked; if (!this.hitSounds) this._cancel(); };
  }

  async load(project) {
    this.project = project;
    this.result = null;
    this.seed = null;
    this._info("");
    try {
      const last = await api.lastGenerated(project.id);        // null: nothing generated yet
      if (last && this.project?.id === project.id) {
        this.result = last;
        this.seed = last.seed;
        $("g-level").value = last.level;
        $("g-slot").value = String(last.slot);
        this._summary();
      }
    } catch (e) { console.warn(e); }
    this.sync();
    this.tl.draw();
  }

  /** Enable controls only with confirmed, unchanged timing. */
  sync() {
    const ok = !!this.project?.timing?.confirmed && !this.timing.dirty;
    $("g-generate").disabled = !ok;
    $("g-reroll").disabled = !ok || !this.result;
    $("g-export").classList.toggle("disabled", !this.result);
    if (this.result) $("g-export").href = api.exportUrl(this.project.id);
    if (!ok && !this.result) this._info("Confirm the timing to generate a chart.");
  }

  async generate({ reroll = false } = {}) {
    const p = this.timing.project ?? this.project;
    this.project = p;
    const level = $("g-level").value, slot = +$("g-slot").value;
    const seed = reroll ? (this.seed ?? 0) + 1 : null;
    $("g-generate").disabled = $("g-reroll").disabled = true;
    this._info("Generating… (the first time also runs the analysis)");
    try {
      this.result = await api.generate(p.id, { level, slot, seed });
      this.seed = this.result.seed;
      this._summary();
    } catch (e) {
      this._info("Generation failed: " + e.message, "bad-text");
    } finally {
      this.sync();
      this._cancel();
      this.tl.draw();
    }
  }

  _summary() {
    const s = this.result.stats, t = this.result.targets;
    const slotName = { 4: "Expert", 5: "Master", 6: "Re:Master" }[this.result.slot];
    const pct = (v) => (v ?? 0).toFixed(0) + "%";
    this._info(`${slotName} ${this.result.level} · ${s.notes} notes · ${s.nps.toFixed(1)} notes/s ` +
      `(target ${t.nps}) · busiest 4 s ${s.peak_nps.toFixed(1)} (${t.peak_nps}) · EACH ${pct(s.each_pct)} ` +
      `· holds ${pct(s.hold_pct)} · slides ${(s.slide_pm ?? 0).toFixed(0)}/min · touch ${pct(s.touch_pct)} ` +
      `· BREAK ${pct(s.break_pct)} · longest stream ${s.longest_stream} · seed ${this.result.seed}`, "ok-text");
  }

  _info(text, cls) {
    const el = $("g-info");
    el.textContent = text;
    el.className = "small " + (cls || "hint");
  }

  // ------------------------------------------------------------ notes lane
  _draw(g, { w, h, tl }) {
    if (!this.result || !this.showNotes) return;
    const lanesTop = h - 3 * 13 - 2;                 // onset lanes start here (analysis.js)
    const y0 = lanesTop - LANE_H - 4;
    g.fillStyle = "#0d0e11cc";
    g.fillRect(0, y0 - 2, w, LANE_H + 4);
    g.font = "10px ui-monospace, Consolas, monospace";
    g.textBaseline = "middle";
    g.fillStyle = "#ff8ad8";
    g.fillText("chart", 4, y0 + LANE_H / 2);
    const t0 = tl.start, t1 = tl.start + tl.viewSec;
    for (const n of this.result.notes) {
      const end = n.end ?? n.t;
      if (end < t0 - 0.1 || n.t > t1 + 0.1) continue;
      const x = Math.round(tl.timeToX(n.t));
      const kinds = n.kinds ?? ["tap"];
      const each = n.buttons.length > 1;
      const color = n.break ? "#ffab40" : kinds.includes("slide") ? "#5ec8ff"
        : kinds.includes("touch") ? "#9dff8a" : each ? "#ffe066" : "#ff8ad8";
      g.fillStyle = color;
      // holds / slides: a thin bar for as long as the hand is busy
      if (end > n.t + 0.01) {
        const x1 = Math.round(tl.timeToX(end));
        g.globalAlpha = 0.55;
        g.fillRect(x, y0 + LANE_H * 0.35, Math.max(1, x1 - x), kinds.includes("slide") ? 3 : 5);
        g.globalAlpha = 1;
      }
      const hh = each ? LANE_H : LANE_H * 0.65;
      g.fillRect(x - 1, y0 + LANE_H - hh, 3, hh);
    }
  }

  // ------------------------------------------------------------ hit sounds
  _cancel() {
    const now = this.player.ctx?.currentTime ?? 0;
    for (const o of this._pending) { try { o.stop(now); } catch {} }
    this._pending = [];
    this._lastCt = 0;
  }

  _tick() {
    const p = this.player;
    if (!this.hitSounds || !this.result || !p.playing || !p.ctx) return;
    if (p.epoch !== this._epoch) { this._cancel(); this._epoch = p.epoch; }
    const now = p.ctx.currentTime, horizon = now + LOOKAHEAD;
    const tNow = p.scheduledTime;
    const notes = this.result.notes;
    let lo = 0, hi = notes.length;                    // first note at/after tNow
    while (lo < hi) { const m = (lo + hi) >> 1; if (notes[m].t < tNow - 0.01) lo = m + 1; else hi = m; }
    this._pending = this._pending.filter((o) => o._end > now);
    for (let i = lo; i < notes.length; i++) {
      const ct = p.ctxTimeAt(notes[i].t);
      if (ct > horizon) break;
      if (ct < now - 0.002 || ct <= this._lastCt + 1e-4) continue;
      this._hit(ct, notes[i].buttons.length > 1 || notes[i].break);
      this._lastCt = ct;
    }
  }

  _hit(when, strong) {
    // short filtered noise burst: a "tap" sound that sits on top of the music
    const ctx = this.player.ctx;
    const len = Math.floor(ctx.sampleRate * 0.03);
    this._noise ??= (() => {
      const b = ctx.createBuffer(1, len, ctx.sampleRate);
      const d = b.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.exp(-i / (len / 5));
      return b;
    })();
    const src = ctx.createBufferSource();
    src.buffer = this._noise;
    const hp = ctx.createBiquadFilter();
    hp.type = "highpass"; hp.frequency.value = 2500;
    const gain = ctx.createGain();
    gain.gain.value = this.hitVolume * (strong ? 1.0 : 0.7);
    src.connect(hp).connect(gain).connect(ctx.destination);
    src.start(when);
    src._end = when + 0.04;
    this._pending.push(src);
  }
}