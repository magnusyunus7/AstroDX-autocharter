// WebAudio player. We decode the whole track into an AudioBuffer and play it
// with AudioBufferSourceNodes, because the audio clock (ctx.currentTime) is
// sample-accurate. That matters in Phase 1.3, when metronome ticks get
// scheduled on the same clock as the music.
//
// Times exposed here are in "analysis time": seconds on the same timeline as
// the server's analysis.wav / waveform. `decoderOffset` (seconds) corrects
// for the browser's mp3 decoder placing audio later (+) or earlier (-) than
// ffmpeg did; see decodercheck.js.

export class Player {
  constructor() {
    this.ctx = null;
    this.buffer = null;
    this.gain = null;
    this.source = null;
    this.playing = false;
    this.startCtxTime = 0;   // ctx.currentTime when playback started
    this.startPos = 0;       // buffer position (s) at that moment
    this.pausedPos = 0;      // buffer position while paused
    this.decoderOffset = 0;  // browser buffer time = analysis time + decoderOffset
    this.onstate = () => {};
  }

  async load(arrayBuffer) {
    this.stop();
    this.ctx ??= new AudioContext({ latencyHint: "interactive" });
    this.gain ??= this.ctx.createGain();
    this.gain.connect(this.ctx.destination);
    this.buffer = await this.ctx.decodeAudioData(arrayBuffer);
    this.pausedPos = 0;
    this.onstate();
    return this.buffer;
  }

  get duration() { return this.buffer ? this.buffer.duration - this.decoderOffset : 0; }

  // Buffer position currently being *heard* (output latency compensated).
  _bufferPos() {
    if (!this.playing) return this.pausedPos;
    const lat = (this.ctx.outputLatency || 0) + (this.ctx.baseLatency || 0);
    const p = this.startPos + (this.ctx.currentTime - this.startCtxTime) - lat;
    return Math.max(this.startPos, p);
  }

  /** Current time in analysis seconds. */
  get time() { return this._bufferPos() - this.decoderOffset; }

  async play(fromTime) {
    if (!this.buffer) return;
    if (this.ctx.state === "suspended") await this.ctx.resume();
    const pos = fromTime !== undefined ? fromTime + this.decoderOffset : this.pausedPos;
    this._stopSource();
    const src = this.ctx.createBufferSource();
    src.buffer = this.buffer;
    src.connect(this.gain);
    const clamped = Math.min(Math.max(pos, 0), this.buffer.duration);
    src.start(0, clamped);
    src.onended = () => {
      if (this.source === src) { this.playing = false; this.pausedPos = this.buffer.duration; this.onstate(); }
    };
    this.source = src;
    this.startPos = clamped;
    this.startCtxTime = this.ctx.currentTime;
    this.playing = true;
    this.onstate();
  }

  pause() {
    if (!this.playing) return;
    this.pausedPos = this._bufferPos();
    this._stopSource();
    this.playing = false;
    this.onstate();
  }

  toggle() { this.playing ? this.pause() : this.play(); }

  seek(time) {
    const pos = Math.min(Math.max(time + this.decoderOffset, 0), this.buffer?.duration ?? 0);
    if (this.playing) this.play(pos - this.decoderOffset);
    else { this.pausedPos = pos; this.onstate(); }
  }

  setVolume(v) { if (this.gain) this.gain.gain.value = v; }

  stop() { this._stopSource(); this.playing = false; this.pausedPos = 0; }

  _stopSource() {
    if (this.source) {
      this.source.onended = null;
      try { this.source.stop(); } catch {}
      this.source.disconnect();
      this.source = null;
    }
  }
}