// Audio helpers - a line-by-line port of voicedetect/audio.py + the pre-processing in voicedetect/detector.py,
// so the browser feeds the model exactly what the Python pipeline would.
export const SR = 16000;

/** librosa.effects.trim(y, top_db): frame RMS (2048 window, 512 hop, zero-padded, centred) vs the loudest frame. */
export function trimSilence(y, topDb = 35, frame = 2048, hop = 512) {
  const nFrames = 1 + Math.floor(y.length / hop);
  const pad = frame >> 1;
  const rms = new Float64Array(nFrames);
  let maxRms = 0;
  for (let f = 0; f < nFrames; f++) {
    let acc = 0;
    const start = f * hop - pad;
    for (let i = 0; i < frame; i++) {
      const j = start + i;
      if (j >= 0 && j < y.length) acc += y[j] * y[j];
    }
    rms[f] = Math.sqrt(acc / frame);
    if (rms[f] > maxRms) maxRms = rms[f];
  }
  const amin = 1e-5;
  const refDb = 20 * Math.log10(Math.max(amin, maxRms));
  let first = -1, last = -1;
  for (let f = 0; f < nFrames; f++) {
    const db = 20 * Math.log10(Math.max(amin, rms[f])) - refDb;
    if (db > -topDb) { if (first < 0) first = f; last = f; }
  }
  if (first < 0) return y.subarray(0, 0);
  return y.subarray(first * hop, Math.min(y.length, (last + 1) * hop));
}

export function padBoth(y, n) {
  const out = new Float32Array(y.length + 2 * n);
  out.set(y, n);
  return out;
}

/** Overlapping fixed windows; returns {wins, starts(in samples)}. Mirrors audio.windows + detector start logic. */
export function windows(y, winS = 3.0, hopS = 1.5, maxWindows = 12) {
  const n = Math.round(winS * SR), hop = Math.round(hopS * SR);
  if (y.length < n) {                                  // short clip: tile it
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = y[i % Math.max(y.length, 1)];
    return { wins: [w], starts: [0] };
  }
  let starts = [];
  for (let s = 0; s <= y.length - n; s += hop) starts.push(s);
  if (starts[starts.length - 1] + n < y.length) starts.push(y.length - n);
  if (starts.length > maxWindows) {                    // np.linspace(0, L-1, max).astype(int), same float ops
    const L = starts.length, step = (L - 1) / (maxWindows - 1), pick = [];
    for (let i = 0; i < maxWindows; i++) pick.push(i === maxWindows - 1 ? L - 1 : Math.floor(i * step));
    starts = pick.map(i => starts[i]);
  }
  return { wins: starts.map(s => y.slice(s, s + n)), starts };
}

function percentile(sorted, q) {                        // numpy 'linear' interpolation
  const idx = (sorted.length - 1) * q, lo = Math.floor(idx), hi = Math.ceil(idx);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
}

/** Fraction of 25 ms frames well above the clip's noise floor (crude energy VAD, as in detector._speech_fraction). */
export function speechFraction(w) {
  const n = Math.round(0.025 * SR), count = Math.floor(w.length / n);
  const e = new Float64Array(count);
  for (let f = 0; f < count; f++) {
    let acc = 0;
    for (let i = 0; i < n; i++) acc += w[f * n + i] * w[f * n + i];
    e[f] = Math.sqrt(acc / n + 1e-12);
  }
  const floor = Math.max(percentile(Float64Array.from(e).sort(), 0.10), 1e-4);
  const thr = Math.max(3.0 * floor, 0.004);
  let above = 0;
  for (const v of e) if (v > thr) above++;
  return above / count;
}

export function peakNormalize(w, peak = 0.95) {
  let m = 0;
  for (let i = 0; i < w.length; i++) { const a = Math.abs(w[i]); if (a > m) m = a; }
  if (m <= 1e-6) return w;
  const out = new Float32Array(w.length), g = peak / m;
  for (let i = 0; i < w.length; i++) out[i] = w[i] * g;
  return out;
}

export function maxAbs(w) {
  let m = 0;
  for (let i = 0; i < w.length; i++) { const a = Math.abs(w[i]); if (a > m) m = a; }
  return m;
}
