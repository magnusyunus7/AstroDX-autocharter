// The music bar: an overview strip (whole song) and a zoomable waveform with
// a time ruler and playhead. All times are in analysis seconds.
//
// Phase 1.3 adds the beat grid on top via `overlays`: functions called with
// (ctx2d, view) after the waveform is drawn.

import { fmtTime } from "./api.js";

const MIN_PX_PER_SEC = 2;
const MAX_PX_PER_SEC = 4000;
const TICK_STEPS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60];

export class Timeline {
  constructor(waveCanvas, overviewCanvas, { getTime, onSeek }) {
    this.wave = waveCanvas;
    this.ov = overviewCanvas;
    this.getTime = getTime;      // () => current playback time
    this.onSeek = onSeek;        // (t) => void
    this.peaks = null;           // Int8Array [min,max,...]
    this.bucketSec = 1;
    this.duration = 0;
    this.start = 0;              // view start (s)
    this.pxPerSec = 100;
    this.follow = true;
    this.playing = false;
    this.markers = [];           // [{t, color, label}]
    this.overlays = [];          // [(g, view) => void]
    this.onShiftDrag = null;     // (dtSec, "start" | "move") => void  (Shift+drag moves the grid)
    this._ovCache = null;
    this._bind();
    new ResizeObserver(() => { this._ovCache = null; this.draw(); }).observe(this.wave);
  }

  setAudio(peaks, meta, duration) {
    this.peaks = peaks;
    this.bucketSec = meta.spp / meta.sample_rate;
    this.duration = duration;
    this._ovCache = null;
    this.pxPerSec = 150;
    this.start = 0;
    this.draw();
  }

  get width() { return this.wave.clientWidth; }
  get viewSec() { return this.width / this.pxPerSec; }
  timeToX(t) { return (t - this.start) * this.pxPerSec; }
  xToTime(x) { return this.start + x / this.pxPerSec; }

  // Zoom around anchorX (px). Default: the playhead if it's on screen, else the center.
  zoom(factor, anchorX) {
    if (anchorX === undefined) {
      const px = this.timeToX(this.getTime());
      anchorX = px >= 0 && px <= this.width ? px : this.width * 0.5;
    }
    const t = this.xToTime(anchorX);
    this.pxPerSec = Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, this.pxPerSec * factor));
    this.start = t - anchorX / this.pxPerSec;
    this._clamp();
    this.draw();
  }

  scrollBy(sec) { this.start += sec; this._clamp(); this.draw(); }

  centerOn(t) { this.start = t - this.viewSec * 0.3; this._clamp(); }

  _clamp() {
    const maxStart = Math.max(0, this.duration - this.viewSec * 0.5);
    this.start = Math.min(Math.max(this.start, -this.viewSec * 0.1), maxStart);
  }

  // ------------------------------------------------------------- drawing
  _fit(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [g, w, h];
  }

  // min/max of peaks between times a..b, scaled to [-1, 1]
  _range(a, b) {
    const n = this.peaks.length >> 1;
    let i0 = Math.max(0, Math.floor(a / this.bucketSec));
    let i1 = Math.min(n, Math.ceil(b / this.bucketSec));
    if (i1 <= i0) { if (i0 >= n) return null; i1 = i0 + 1; }
    let mn = 127, mx = -127;
    for (let i = i0; i < i1; i++) {
      const lo = this.peaks[2 * i], hi = this.peaks[2 * i + 1];
      if (lo < mn) mn = lo;
      if (hi > mx) mx = hi;
    }
    return [mn / 127, mx / 127];
  }

  draw() {
    if (this.playing && this.follow) {
      const t = this.getTime();
      if (t < this.start || t > this.start + this.viewSec * 0.7) this.centerOn(t);
      // smooth follow: keep playhead at 30% once it gets there
      else if (t > this.start + this.viewSec * 0.3) this.start = t - this.viewSec * 0.3;
    }
    this._drawWave();
    this._drawOverview();
  }

  _drawWave() {
    const [g, w, h] = this._fit(this.wave);
    if (!w || !h) return;  // music bar hidden
    g.clearRect(0, 0, w, h);
    const rulerH = 18, mid = rulerH + (h - rulerH) / 2, amp = (h - rulerH) / 2 - 4;

    // ruler + grid
    const step = TICK_STEPS.find((s) => s * this.pxPerSec >= 80) ?? 60;
    g.font = "11px ui-monospace, Consolas, monospace";
    g.textBaseline = "top";
    const t0 = Math.floor(this.start / step) * step;
    for (let t = t0; t < this.start + this.viewSec; t += step) {
      const x = Math.round(this.timeToX(t)) + 0.5;
      g.strokeStyle = "#262933";
      g.beginPath(); g.moveTo(x, rulerH); g.lineTo(x, h); g.stroke();
      g.strokeStyle = "#555b6b";
      g.beginPath(); g.moveTo(x, rulerH - 6); g.lineTo(x, rulerH); g.stroke();
      g.fillStyle = "#8b8f9c";
      g.fillText(step < 1 ? fmtTime(t) : fmtTime(t).slice(0, 5), x + 3, 2);
    }
    g.strokeStyle = "#333744";
    g.beginPath(); g.moveTo(0, rulerH + 0.5); g.lineTo(w, rulerH + 0.5); g.stroke();

    // waveform
    if (this.peaks) {
      g.fillStyle = "#4fc3f7";
      const secPerPx = 1 / this.pxPerSec;
      for (let x = 0; x < w; x++) {
        const a = this.xToTime(x);
        if (a + secPerPx < 0 || a > this.duration) continue;
        const r = this._range(Math.max(a, 0), a + secPerPx);
        if (!r) continue;
        const y1 = mid - r[1] * amp, y2 = mid - r[0] * amp;
        g.fillRect(x, y1, 1, Math.max(1, y2 - y1));
      }
      // song end
      const xe = this.timeToX(this.duration);
      if (xe < w) { g.fillStyle = "#00000088"; g.fillRect(xe, rulerH, w - xe, h - rulerH); }
    }

    for (const fn of this.overlays) fn(g, { w, h, rulerH, tl: this });

    for (const m of this.markers) {
      const x = Math.round(this.timeToX(m.t)) + 0.5;
      if (x < 0 || x > w) continue;
      g.strokeStyle = m.color; g.setLineDash([4, 3]);
      g.beginPath(); g.moveTo(x, rulerH); g.lineTo(x, h); g.stroke();
      g.setLineDash([]);
      if (m.label) { g.fillStyle = m.color; g.fillText(m.label, x + 3, h - 14); }
    }

    // playhead
    const px = Math.round(this.timeToX(this.getTime())) + 0.5;
    if (px >= 0 && px <= w) {
      g.strokeStyle = "#ffd54f"; g.lineWidth = 2;
      g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
      g.lineWidth = 1;
    }
  }

  _drawOverview() {
    const [g, w, h] = this._fit(this.ov);
    if (!w || !h) return;  // music bar hidden
    if (!this.peaks || !this.duration) { g.clearRect(0, 0, w, h); return; }
    if (!this._ovCache || this._ovCache.width !== this.ov.width) {
      // the whole-song waveform only changes on resize; cache it
      const c = document.createElement("canvas");
      c.width = this.ov.width; c.height = this.ov.height;
      const cg = c.getContext("2d");
      const dpr = window.devicePixelRatio || 1;
      cg.setTransform(dpr, 0, 0, dpr, 0, 0);
      cg.fillStyle = "#2f6f8f";
      const mid = h / 2, amp = h / 2 - 2, secPerPx = this.duration / w;
      for (let x = 0; x < w; x++) {
        const r = this._range(x * secPerPx, (x + 1) * secPerPx);
        if (!r) continue;
        cg.fillRect(x, mid - r[1] * amp, 1, Math.max(1, (r[1] - r[0]) * amp));
      }
      this._ovCache = c;
    }
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, this.ov.width, this.ov.height);
    g.drawImage(this._ovCache, 0, 0);
    const dpr = window.devicePixelRatio || 1;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const k = w / this.duration;
    g.fillStyle = "#ffffff18"; g.strokeStyle = "#ffffff55";
    g.fillRect(this.start * k, 0, this.viewSec * k, h);
    g.strokeRect(this.start * k + 0.5, 0.5, this.viewSec * k, h - 1);
    g.fillStyle = "#ffd54f";
    g.fillRect(this.getTime() * k - 1, 0, 2, h);
  }

  // ------------------------------------------------------------ input
  _bind() {
    const wv = this.wave;
    wv.addEventListener("wheel", (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        this.zoom(Math.exp(-e.deltaY * 0.002), e.offsetX);
      } else {
        const d = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY;
        this.scrollBy(d / this.pxPerSec);
      }
    }, { passive: false });

    let drag = null;
    wv.addEventListener("pointerdown", (e) => {
      wv.setPointerCapture(e.pointerId);
      const grid = e.shiftKey && this.onShiftDrag;
      drag = { x: e.offsetX, start: this.start, moved: false, grid };
      if (grid) this.onShiftDrag(0, "start");
    });
    wv.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.offsetX - drag.x;
      if (Math.abs(dx) > 3) { drag.moved = true; wv.classList.add("dragging"); }
      if (!drag.moved) return;
      if (drag.grid) this.onShiftDrag(dx / this.pxPerSec, "move");
      else { this.start = drag.start - dx / this.pxPerSec; this._clamp(); this.draw(); }
    });
    wv.addEventListener("pointerup", (e) => {
      if (drag && !drag.moved) this.onSeek(Math.max(0, Math.min(this.duration, this.xToTime(e.offsetX))));
      drag = null; wv.classList.remove("dragging");
    });

    // overview: click / drag to move the view (and seek when not playing)
    let ovDrag = false;
    const ovMove = (e) => {
      const t = (e.offsetX / this.ov.clientWidth) * this.duration;
      this.start = t - this.viewSec / 2; this._clamp(); this.draw();
    };
    this.ov.addEventListener("pointerdown", (e) => { ovDrag = true; this.ov.setPointerCapture(e.pointerId); ovMove(e); });
    this.ov.addEventListener("pointermove", (e) => { if (ovDrag) ovMove(e); });
    this.ov.addEventListener("pointerup", () => { ovDrag = false; });
  }
}