// Measures how far the browser's mp3 decode is shifted relative to the
// server's (ffmpeg) decode, by cross-correlating loudness envelopes.
//
// Server envelope: from peaks.bin, one value per `spp` samples at `sample_rate`.
// Browser envelope: same bucket duration, computed from the AudioBuffer.
// Result: offset in seconds, positive = browser audio is LATER.

const MAX_LAG_SEC = 0.12;   // search +-120 ms
const WINDOW_SEC = 60;      // compare up to 60 s of audio after the first sound

export function measureDecoderOffset(audioBuffer, peaks, meta) {
  const bucketSec = meta.spp / meta.sample_rate;

  // server envelope
  const nSrv = peaks.length >> 1;
  const srv = new Float32Array(nSrv);
  for (let i = 0; i < nSrv; i++) srv[i] = Math.max(-peaks[2 * i], peaks[2 * i + 1]);

  // browser envelope (mono, abs max per bucket)
  const sr = audioBuffer.sampleRate;
  const chans = [];
  for (let c = 0; c < audioBuffer.numberOfChannels; c++) chans.push(audioBuffer.getChannelData(c));
  const nBr = Math.floor(audioBuffer.duration / bucketSec);
  const br = new Float32Array(nBr);
  for (let i = 0; i < nBr; i++) {
    const a = Math.floor(i * bucketSec * sr), b = Math.floor((i + 1) * bucketSec * sr);
    let m = 0;
    for (let s = a; s < b; s++) {
      let v = 0;
      for (const ch of chans) v += ch[s];
      v = Math.abs(v / chans.length);
      if (v > m) m = v;
    }
    br[i] = m;
  }

  const start = Math.max(0, Math.floor((meta.first_sound || 0) / bucketSec));
  const len = Math.min(Math.floor(WINDOW_SEC / bucketSec), nSrv - start, nBr - start);
  const maxLag = Math.ceil(MAX_LAG_SEC / bucketSec);
  if (len < 200) return { ok: false, reason: "audio too short to check" };

  const norm = (arr, off) => {
    const out = new Float32Array(len + 2 * maxLag);
    let mean = 0, n = 0;
    for (let i = -maxLag; i < len + maxLag; i++) {
      const j = off + i;
      const v = j >= 0 && j < arr.length ? arr[j] : 0;
      out[i + maxLag] = v; mean += v; n++;
    }
    mean /= n;
    let sd = 0;
    for (let i = 0; i < out.length; i++) { out[i] -= mean; sd += out[i] * out[i]; }
    sd = Math.sqrt(sd / out.length) || 1;
    for (let i = 0; i < out.length; i++) out[i] /= sd;
    return out;
  };
  const S = norm(srv, start), B = norm(br, start);

  // corr(lag) = sum S[i] * B[i + lag]; positive lag -> browser later
  const corr = [];
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let sum = 0;
    for (let i = maxLag; i < maxLag + len; i++) sum += S[i] * B[i + lag];
    corr.push(sum / len);
  }
  let best = 0;
  for (let k = 1; k < corr.length; k++) if (corr[k] > corr[best]) best = k;

  // parabolic interpolation for sub-bucket precision
  let frac = 0;
  if (best > 0 && best < corr.length - 1) {
    const [y0, y1, y2] = [corr[best - 1], corr[best], corr[best + 1]];
    const d = y0 - 2 * y1 + y2;
    if (d !== 0) frac = 0.5 * (y0 - y2) / d;
  }
  const lagBuckets = best - maxLag + frac;
  return {
    ok: corr[best] > 0.5,          // weak correlation -> don't trust it
    offset: lagBuckets * bucketSec,
    confidence: corr[best],
  };
}