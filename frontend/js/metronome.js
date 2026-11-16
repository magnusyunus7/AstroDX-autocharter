// Metronome: clicks on every beat of the grid while the music plays.
//
// Uses the standard WebAudio "lookahead scheduler": a timer wakes up every
// 25 ms and schedules the clicks falling in the next ~120 ms at exact
// AudioContext times, using player.ctxTimeAt(). Music and clicks therefore
// share one sample-accurate clock, so if the clicks sound early/late against
// the kicks, the *grid* is off, not the metronome.

const LOOKAHEAD = 0.12;   // seconds scheduled ahead
const INTERVAL = 25;      // ms between scheduler runs

export class Metronome {
  constructor(player, getGrid) {
    this.player = player;
    this.getGrid = getGrid;   // () => { bpm, first, beatsPerBar }
    this.enabled = false;
    this.volume = 0.6;
    this._epoch = -1;
    this._lastCt = 0;         // ctx time of the last scheduled click
    this._pending = [];       // scheduled oscillators (so we can cancel)
    this._gridKey = "";
    setInterval(() => this._tick(), INTERVAL);
  }

  setEnabled(on) { this.enabled = on; if (!on) this._cancel(); }

  _cancel() {
    const now = this.player.ctx?.currentTime ?? 0;
    for (const o of this._pending) { try { o.stop(now); } catch {} }
    this._pending = [];
    this._lastCt = 0;
  }

  _tick() {
    const p = this.player;
    if (!this.enabled || !p.playing || !p.ctx) return;
    const g = this.getGrid();
    const key = `${g.bpm}|${g.first}`;
    // restart/seek/speed change or grid edit: drop already-scheduled clicks
    if (p.epoch !== this._epoch || key !== this._gridKey) {
      this._cancel();
      this._epoch = p.epoch;
      this._gridKey = key;
    }
    const ctx = p.ctx, now = ctx.currentTime, horizon = now + LOOKAHEAD;
    const beat = 60 / g.bpm;
    const tNow = p.scheduledTime;
    let k = Math.ceil((tNow - g.first) / beat - 1e-6);
    this._pending = this._pending.filter((o) => o._end > now);
    for (;; k++) {
      const t = g.first + k * beat;
      const ct = p.ctxTimeAt(t);
      if (ct > horizon) break;
      if (t < 0 || ct < now - 0.002 || ct <= this._lastCt + 1e-4) continue;
      const bar = ((k % g.beatsPerBar) + g.beatsPerBar) % g.beatsPerBar === 0;
      this._click(ct, bar);
      this._lastCt = ct;
    }
  }

  _click(when, accent) {
    const ctx = this.player.ctx;
    const osc = ctx.createOscillator();
    const env = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = accent ? 1760 : 1175;
    const v = this.volume * (accent ? 0.5 : 0.32);
    env.gain.setValueAtTime(v, when);
    env.gain.exponentialRampToValueAtTime(0.0001, when + 0.035);
    osc.connect(env).connect(ctx.destination);
    osc.start(when);
    osc.stop(when + 0.04);
    osc._end = when + 0.04;
    this._pending.push(osc);
  }
}