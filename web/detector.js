// In-browser detector: same logic as voicedetect/detector.py (trim -> windows -> VAD -> WavLM features -> linear classifier
// -> logit-space aggregation -> verdict). Runs entirely on the visitor's device.
import * as ort from './ort/ort.wasm.min.mjs';
import { SR, trimSilence, padBoth, windows, speechFraction, peakNormalize, maxAbs } from './dsp.js';

const MIN_SECONDS = 1.0, MAX_SECONDS = 60.0;
const clamp = p => Math.min(1 - 1e-4, Math.max(1e-4, p));
const logit = p => { p = clamp(p); return Math.log(p / (1 - p)); };
const sigmoid = z => 1 / (1 + Math.exp(-z));

export class AudioError extends Error {}

async function fetchBytes(url, onProgress, expected) {
  const res = await fetch(url);
  if (!res.ok) throw new Error('Could not download ' + url + ' (' + res.status + ')');
  const total = Number(res.headers.get('content-length')) || expected || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (onProgress) onProgress(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

export class Detector {
  static async load(onProgress, base = '.') {
    ort.env.wasm.wasmPaths = new URL(base + '/ort/', location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1)) : 1;
    const cls = await (await fetch(base + '/model/classifier.json', { cache: 'no-cache' })).json();
    const bytes = await fetchBytes(base + '/model/' + cls.model_file, onProgress, cls.model_bytes);
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    let head = null;
    if (cls.type === 'mlp') {                              // MLP head weights (scaler already folded in), little-endian float32
      const raw = await fetchBytes(base + '/model/' + cls.head_file, null, cls.head_bytes);
      head = new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
    }
    return new Detector(session, cls, head);
  }

  constructor(session, cls, head) {
    this.session = session;
    this.cls = cls;
    this.winN = Math.round(cls.window_s * SR);
    if (cls.type === 'mlp') {
      const H = cls.hidden, nIn = cls.n_in;
      this.H = H; this.nIn = nIn;
      this.W1 = head.subarray(0, nIn * H);                 // row-major (nIn x H)
      this.b1 = head.subarray(nIn * H, nIn * H + H);
      this.w2 = head.subarray(nIn * H + H, nIn * H + 2 * H);
      this.b2 = head[nIn * H + 2 * H];
    } else {
      this.w = Float32Array.from(cls.weights);
    }
    this.realMax = cls.real_max ?? 0.5 - cls.unsure_margin;   // verdict band on the calibrated P(AI)
    this.aiMin = cls.ai_min ?? 0.5 + cls.unsure_margin;
  }

  /** Classifier logit for one pooled-feature vector. */
  logitOf(f) {
    if (this.cls.type !== 'mlp') {
      let z = this.cls.bias;
      for (let i = 0; i < this.w.length; i++) z += this.w[i] * f[i];
      return z;
    }
    const { H, nIn, W1 } = this;
    const h = Float32Array.from(this.b1);
    for (let i = 0; i < nIn; i++) {
      const xi = f[i];
      if (xi === 0) continue;
      const row = i * H;
      for (let j = 0; j < H; j++) h[j] += xi * W1[row + j];
    }
    let z = this.b2;
    for (let j = 0; j < H; j++) if (h[j] > 0) z += h[j] * this.w2[j];
    return z;
  }

  /** P(AI) for one 3 s window (before threshold shift). */
  async windowProb(w) {
    const x = new ort.Tensor('float32', peakNormalize(w), [1, this.winN]);
    const out = await this.session.run({ audio: x });
    const f = out[Object.keys(out)[0]].data;               // (1, nLayers, 1536) flattened, layer-major like the classifier
    const z = this.logitOf(f);
    return { p: sigmoid(z), z };
  }

  /** y: Float32Array mono 16 kHz. Returns the same structure as the Python/Flask API. */
  async predictWave(y, onWindow) {
    const cls = this.cls;
    let dur = y.length / SR;
    if (dur < MIN_SECONDS) throw new AudioError('The audio is too short. Please provide at least 1-2 seconds of speech.');
    if (dur > MAX_SECONDS) { y = y.subarray(0, MAX_SECONDS * SR); }
    for (let i = 0; i < y.length; i += 997) if (!Number.isFinite(y[i])) throw new AudioError('The audio contains invalid samples.');
    const t = trimSilence(y, 35);                          // training clips were trimmed the same way, never below 1 s
    if (t.length >= MIN_SECONDS * SR) { y = padBoth(t, Math.round(0.1 * SR)); dur = y.length / SR; }

    const { wins, starts } = windows(y, cls.window_s, cls.hop_s, cls.max_windows);
    const speech = wins.map(w => speechFraction(w) >= 0.15 && maxAbs(w) > 0.003);
    if (!speech.some(Boolean)) throw new AudioError('No speech detected - the recording seems silent or too quiet.');

    const thrLogit = logit(cls.threshold);
    const pw = wins.map((_, i) => ({ start: starts[i] / SR, skipped: true, prob_ai: 0 }));
    const live = [];
    let done = 0; const total = speech.filter(Boolean).length;
    for (let i = 0; i < wins.length; i++) {
      if (!speech[i]) continue;
      const { p } = await this.windowProb(wins[i]);
      live.push(logit(p));
      pw[i].skipped = false;
      pw[i].prob_ai = sigmoid(logit(p) - thrLogit);
      if (onWindow) onWindow(++done, total);
    }
    const z = live.reduce((a, b) => a + b, 0) / live.length - thrLogit;
    const pAi = sigmoid(z);
    const lo = this.realMax, hi = this.aiMin;
    const verdict = pAi >= hi ? 'ai' : pAi <= lo ? 'real' : 'unsure';

    const notes = [];
    if (dur < 5) notes.push('Short clip: accuracy improves a lot with 6-10+ seconds of continuous speech.');
    if (speech.some(s => !s)) notes.push(speech.filter(s => !s).length + ' silent segment(s) were ignored.');
    const ps = pw.filter(w => !w.skipped).map(w => w.prob_ai);
    if (ps.length > 1 && Math.max(...ps) >= hi && Math.min(...ps) <= lo)
      notes.push('Segments disagree: part of the clip looks real and part looks AI. It may be spliced or mixed.');
    if (verdict === 'unsure') notes.push('The model is not confident. Try a longer, cleaner recording.');
    notes.push('Heavily compressed audio (e.g. voice messages, phone calls) is harder to judge reliably.');
    return { verdict, prob_ai: pAi, duration: dur, n_windows: wins.length, windows: pw, notes };
  }
}

/** Decode any browser-supported audio (wav/mp3/m4a/ogg/flac/webm) to mono 16 kHz Float32Array. */
export async function decodeToMono16k(arrayBuffer) {
  const AC = window.AudioContext || window.webkitAudioContext;
  let ctx, buf;
  try { ctx = new AC({ sampleRate: SR }); buf = await ctx.decodeAudioData(arrayBuffer.slice(0)); }
  catch (e) {
    if (ctx) ctx.close();
    ctx = new AC();
    try { buf = await ctx.decodeAudioData(arrayBuffer.slice(0)); }
    catch (e2) { ctx.close(); throw new AudioError('Could not read this audio file. Try WAV, MP3, M4A, FLAC, OGG or WebM.'); }
  }
  ctx.close();
  if (buf.sampleRate !== SR) {                            // fallback path: resample offline
    const off = new OfflineAudioContext(buf.numberOfChannels, Math.ceil(buf.duration * SR), SR);
    const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
    buf = await off.startRendering();
  }
  const out = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += ch[i] / buf.numberOfChannels;
  }
  return out;
}
