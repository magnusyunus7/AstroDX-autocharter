// Timing controls: BPM, offset (time of a downbeat), beat grid drawing,
// metronome, playback speed, tap tempo, and saving to the project.
//
// `first` is the time of ANY bar's beat 1. Export moves it back to the
// first bar that fits in the song, so only its phase matters.

import { api, fmtTime } from "./api.js";
import { Metronome } from "./metronome.js";

const BEATS_PER_BAR = 4;
const $ = (id) => document.getElementById(id);

export class TimingControls {
  constructor(player, timeline) {
    this.player = player;
    this.tl = timeline;
    this.project = null;
    this.bpm = 120;
    this.first = 0;
    this.saved = null;     // {bpm, first} as last saved/confirmed
    this.taps = [];
    this.metronome = new Metronome(player, () => ({ bpm: this.bpm, first: this.first, beatsPerBar: BEATS_PER_BAR }));

    timeline.overlays.push((g, v) => this._drawGrid(g, v));
    timeline.onShiftDrag = (dt, phase) => {
      if (phase === "start") this._dragFirst = this.first;
      else this.setFirst(this._dragFirst + dt);
    };
    this._bindUI();
  }

  get beat() { return 60 / this.bpm; }
  get bar() { return this.beat * BEATS_PER_BAR; }
  get dirty() { return !this.saved || this.saved.bpm !== this.bpm || this.saved.first !== this.first; }

  load(project, meta) {
    this.project = project;
    if (project.timing) {
      this.bpm = project.timing.bpm;
      this.first = project.timing.first;
      this.saved = project.timing.confirmed ? { bpm: this.bpm, first: this.first } : null;
    } else {
      // No detection until Phase 1.4: start at 120 BPM with beat 1 on the first sound.
      this.bpm = 120;
      this.first = meta.first_sound || 0;
      this.saved = null;
    }
    this._refresh();
  }

  // ------------------------------------------------------------ setters
  setBpm(bpm) {
    if (!isFinite(bpm) || bpm <= 10 || bpm > 1000) return;
    this.bpm = Math.round(bpm * 10000) / 10000;
    this._refresh();
  }

  setFirst(t) {
    if (!isFinite(t)) return;
    // keep it inside the song by wrapping whole bars (same grid, different bar 1)
    while (t < 0) t += this.bar;
    const dur = this.project?.duration ?? Infinity;
    while (t >= dur && t - this.bar >= 0) t -= this.bar;
    this.first = Math.round(t * 10000) / 10000;
    this._refresh();
  }

  nudge(sec) { this.setFirst(this.first + sec); }

  /** Make the playhead position a downbeat. */
  firstFromPlayhead() { this.setFirst(this.player.time); }

  tap() {
    const now = performance.now() / 1000;
    if (this.taps.length && now - this.taps[this.taps.length - 1] > 2) this.taps = [];
    this.taps.push(now);
    if (this.taps.length > 9) this.taps.shift();
    const n = this.taps.length;
    $("tap-info").textContent = n < 2 ? "tap…" : `${n} taps`;
    if (n >= 4) {
      const avg = (this.taps[n - 1] - this.taps[0]) / (n - 1);
      this.setBpm(Math.round(60 / avg * 100) / 100);
    }
  }

  async save() {
    const btn = $("btn-confirm");
    btn.disabled = true;
    try {
      this.project = await api.putTiming(this.project.id, { bpm: this.bpm, first: this.first, confirmed: true });
      this.saved = { bpm: this.bpm, first: this.first };
    } catch (e) {
      alert("Couldn't save timing: " + e.message);
    } finally {
      btn.disabled = false;
      this._refresh();
    }
  }

  // ------------------------------------------------------------ UI
  _bindUI() {
    const bpm = $("t-bpm"), first = $("t-first");
    bpm.addEventListener("change", () => this.setBpm(parseFloat(bpm.value)));
    first.addEventListener("change", () => this.setFirst(parseFloat(first.value)));
    $("t-x2").onclick = () => this.setBpm(this.bpm * 2);
    $("t-half").onclick = () => this.setBpm(this.bpm / 2);
    for (const btn of document.querySelectorAll("[data-nudge]")) {
      btn.onclick = () => this.nudge(parseFloat(btn.dataset.nudge) / 1000);
    }
    $("t-here").onclick = () => this.firstFromPlayhead();
    $("t-tap").onclick = () => this.tap();
    $("t-metro").onchange = (e) => this.metronome.setEnabled(e.target.checked);
    $("t-metro-vol").oninput = (e) => { this.metronome.volume = +e.target.value; };
    $("t-speed").onchange = (e) => this.player.setRate(parseFloat(e.target.value));
    $("btn-confirm").onclick = () => this.save();

    document.addEventListener("keydown", (e) => {
      if ($("view-project").hidden || e.target.tagName === "INPUT" && e.target.type !== "checkbox"
          && e.target.type !== "range") return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "m") { const c = $("t-metro"); c.checked = !c.checked; c.onchange({ target: c }); }
      else if (k === "t") this.tap();
      else if (k === "o") this.firstFromPlayhead();
      else if (e.key === "[" || e.key === "{") this.nudge(e.shiftKey ? -0.01 : -0.001);
      else if (e.key === "]" || e.key === "}") this.nudge(e.shiftKey ? 0.01 : 0.001);
      else return;
      e.preventDefault();
    });
  }

  _refresh() {
    const bpm = $("t-bpm"), first = $("t-first");
    if (document.activeElement !== bpm) bpm.value = this.bpm;
    if (document.activeElement !== first) first.value = this.first.toFixed(4);
    const st = $("t-status");
    if (this.saved && !this.dirty) {
      st.textContent = "✓ confirmed"; st.className = "status ok";
    } else {
      st.textContent = this.saved ? "● unsaved changes" : "● not confirmed"; st.className = "status warn";
    }
    $("timing-summary").textContent =
      `BPM ${this.bpm} · beat ${(this.beat * 1000).toFixed(1)} ms · bar ${(this.bar * 1000).toFixed(1)} ms · ` +
      `downbeat at ${fmtTime(this.first)}`;
    this.tl.draw();
  }

  _drawGrid(g, { w, h, rulerH, tl }) {
    const beat = this.beat;
    const pxBeat = beat * tl.pxPerSec;
    if (pxBeat * BEATS_PER_BAR < 6) return;            // too zoomed out to be useful
    const sub = pxBeat > 90 ? 4 : pxBeat > 40 ? 2 : 1;  // subdivisions when zoomed in
    const step = beat / sub;
    const t0 = Math.max(0, tl.start), t1 = Math.min(tl.duration, tl.start + tl.viewSec);
    let k = Math.ceil((t0 - this.first) / step - 1e-9);
    g.font = "11px ui-monospace, Consolas, monospace";
    g.textBaseline = "top";
    for (;; k++) {
      const t = this.first + k * step;
      if (t > t1) break;
      const x = Math.round(tl.timeToX(t)) + 0.5;
      const isBeat = k % sub === 0;
      const beatIdx = Math.round(k / sub);
      const isBar = isBeat && ((beatIdx % BEATS_PER_BAR) + BEATS_PER_BAR) % BEATS_PER_BAR === 0;
      if (!isBeat && pxBeat < 40) continue;
      if (!isBar && pxBeat < 8) continue;
      g.strokeStyle = isBar ? "#ff4f86" : isBeat ? "#ffffffb0" : "#ffffff40";
      g.lineWidth = isBar ? 2 : 1;
      g.beginPath(); g.moveTo(x, rulerH); g.lineTo(x, h); g.stroke();
      if (isBar) {
        // bar number tag just under the ruler
        const label = String(Math.floor(beatIdx / BEATS_PER_BAR) + 1);
        const tw = g.measureText(label).width + 6;
        g.fillStyle = "#ff4f86";
        g.fillRect(x, rulerH, tw, 14);
        g.fillStyle = "#0d0e11";
        g.fillText(label, x + 3, rulerH + 1);
      }
    }
    g.lineWidth = 1;
  }
}