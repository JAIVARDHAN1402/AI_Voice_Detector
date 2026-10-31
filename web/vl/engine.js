// VoiceLock engine that runs entirely in the browser (a port of the Python package `voicelock/`).
//
//  Layer 1  AI-voice forensics     : the site's WavLM detector (../detector.js) + acoustic checks
//  Layer 2  speaker identity       : ECAPA-TDNN (ONNX, same weights as SpeechBrain) cosine to the enrolled voiceprint + segment consistency
//  Layer 3  transaction challenge  : phrase derived with HMAC-SHA256 from (transaction hash, session, nonce, pass), checked with Whisper tiny.en
//  Layer 4  risk + authorisation   : all layers must pass; risk tier tightens thresholds; critical = two challenges; signed single-use token
//
// IMPORTANT: everything (keys, secret, sessions, audit log) lives in THIS browser, so this build is a working demonstration of the
// protocol, not a security boundary. A real deployment keeps the secret/keys/state on a server (the Python package does that).
import * as ort from '../ort/ort.wasm.min.mjs';
import { Detector, AudioError, decodeToMono16k } from '../detector.js';
import { trimSilence, speechFraction } from '../dsp.js';
import { fetchRobust } from './loader.js';

export const SR = 16000;
export const CFG = {
  challengeTtl: 90, testChallengeTtl: 900, sessionTtl: 300, tokenTtl: 60, maxAttempts: 3, maxAudio: 20, minAudio: 1.5, minEnroll: 3,
  challengeMinMatch: 6 / 7,
  aiLimit: { low: 0.50, medium: 0.45, high: 0.35, critical: 0.25 }, aiSevere: 0.90,
  speakerMin: { low: 0.40, medium: 0.45, high: 0.50, critical: 0.55 }, speakerSevere: 0.15, consistencyMin: 0.35,
  amountTiers: [10000, 100000, 1000000], requiredPasses: { low: 1, medium: 1, high: 1, critical: 2 },
  offHours: [8, 20], velocityWindow: 3600, velocityLimit: 3,
};
const TIERS = ['low', 'medium', 'high', 'critical'];
const TIER_RISK = { low: 0.15, medium: 0.40, high: 0.70, critical: 0.95 };
const WORDS = `amber anchor apple arrow basket beacon bridge cactus candle canyon carpet castle cobalt copper coral crystal
dolphin eagle ember falcon forest garden glacier harbor hazel island jacket jungle kettle lantern lemon marble meadow
mirror nectar orchid oyster palace pepper pillow planet pocket rabbit river saddle silver spider summer temple thunder
tiger velvet violet walnut window winter yellow zebra anvil bamboo canvas desert engine feather gadget helmet`.split(/\s+/);
const DIGIT_WORDS = { zero: '0', oh: '0', one: '1', two: '2', three: '3', four: '4', five: '5', six: '6', seven: '7', eight: '8', nine: '9' };
export const TEST_PHRASE = { words: ['amber', 'candle', 'river'], digits: '1234' };

export class VLError extends Error { constructor(msg, status = 400) { super(msg); this.status = status; } }

// ------------------------------------------------------------------------------------------------ small helpers
const enc = new TextEncoder();
const hex = b => [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, '0')).join('');
const sha256 = async d => new Uint8Array(await crypto.subtle.digest('SHA-256', typeof d === 'string' ? enc.encode(d) : d));
const b64u = bytes => { let s = ''; bytes = new Uint8Array(bytes); for (let i = 0; i < bytes.length; i += 8192) s += String.fromCharCode(...bytes.subarray(i, i + 8192)); return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''); };
const b64uBytes = s => { s = s.replace(/-/g, '+').replace(/_/g, '/'); while (s.length % 4) s += '='; return Uint8Array.from(atob(s), c => c.charCodeAt(0)); };
const randHex = n => hex(crypto.getRandomValues(new Uint8Array(n)));
const stable = v => v && typeof v === 'object' && !Array.isArray(v)
  ? '{' + Object.keys(v).sort().map(k => JSON.stringify(k) + ':' + stable(v[k])).join(',') + '}'
  : Array.isArray(v) ? '[' + v.map(stable).join(',') + ']' : JSON.stringify(v);
const now = () => Date.now() / 1000;
const unit = v => { let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n) + 1e-9; return Float32Array.from(v, x => x / n); };
const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
const f32ToB64 = a => b64u(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const b64ToF32 = s => { const u = b64uBytes(s); return new Float32Array(u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength)); };

function pctl(arr, p) {                                   // numpy-style linear interpolation
  const s = Float64Array.from(arr).sort(); const k = (s.length - 1) * p / 100, f = Math.floor(k), c = Math.min(f + 1, s.length - 1);
  return s[f] + (s[c] - s[f]) * (k - f);
}

// ------------------------------------------------------------------------------------------------ transaction canonicalisation
function quantize2(a) {                                    // Decimal.quantize(0.01), ROUND_HALF_EVEN, no floating point
  const m = String(a).trim().match(/^([+-]?)(\d+)(?:\.(\d*))?$/);
  if (!m) throw new VLError('invalid transaction: invalid amount');
  const frac = m[3] || ''; let q = BigInt(m[2] + frac); const scale = frac.length;
  if (scale <= 2) q *= 10n ** BigInt(2 - scale);
  else { const div = 10n ** BigInt(scale - 2), r = q % div; q /= div; const half = div / 2n; if (r > half || (r === half && q % 2n === 1n)) q += 1n; }
  if (m[1] === '-' || q <= 0n) throw new VLError('invalid transaction: amount must be positive');
  return `${q / 100n}.${String(q % 100n).padStart(2, '0')}`;
}
export function canonicalTxn(t) {
  return { user_id: String(t.user_id || '').trim(), amount: quantize2(t.amount), currency: String(t.currency || 'INR').trim().toUpperCase(),
           recipient: String(t.recipient || '').trim().replace(/\s+/g, ' ').toLowerCase(), reference: String(t.reference || '').trim() };
}
export const txnHash = async t => hex(await sha256(stable(canonicalTxn(t))));

// ------------------------------------------------------------------------------------------------ keys, tokens
let KEYS = null;
async function keys() {
  if (KEYS) return KEYS;
  const saved = (() => { try { return JSON.parse(localStorage.getItem('vl.keys') || 'null'); } catch (e) { return null; } })();
  const algos = [{ name: 'Ed25519', alg: 'EdDSA' }, { name: 'ECDSA', namedCurve: 'P-256', alg: 'ES256', hash: 'SHA-256' }];
  let a, priv, pub;
  if (saved) {
    a = algos.find(x => x.alg === saved.alg);
    try { priv = await crypto.subtle.importKey('jwk', saved.priv, a.name === 'ECDSA' ? { name: 'ECDSA', namedCurve: 'P-256' } : { name: a.name }, true, ['sign']);
          pub = await crypto.subtle.importKey('jwk', saved.pub, a.name === 'ECDSA' ? { name: 'ECDSA', namedCurve: 'P-256' } : { name: a.name }, true, ['verify']); }
    catch (e) { a = null; }
  }
  if (!a || !priv) {
    for (const x of algos) {
      try {
        const kp = await crypto.subtle.generateKey(x.name === 'ECDSA' ? { name: 'ECDSA', namedCurve: 'P-256' } : { name: x.name }, true, ['sign', 'verify']);
        a = x; priv = kp.privateKey; pub = kp.publicKey; break;
      } catch (e) { /* try the next algorithm */ }
    }
    if (!a) throw new VLError('This browser has no WebCrypto signing support (needs Ed25519 or ECDSA).', 500);
    try { localStorage.setItem('vl.keys', JSON.stringify({ alg: a.alg, priv: await crypto.subtle.exportKey('jwk', priv), pub: await crypto.subtle.exportKey('jwk', pub) })); } catch (e) { /* memory only */ }
  }
  let secret = null;
  try { secret = localStorage.getItem('vl.secret'); } catch (e) { /* ignore */ }
  if (!secret) { secret = randHex(32); try { localStorage.setItem('vl.secret', secret); } catch (e) { /* ignore */ } }
  const jwk = await crypto.subtle.exportKey('jwk', pub);
  const kid = hex(await sha256((jwk.x || '') + (jwk.y || ''))).slice(0, 16);
  const signParams = a.name === 'ECDSA' ? { name: 'ECDSA', hash: 'SHA-256' } : { name: a.name };
  KEYS = { alg: a.alg, priv, pub, kid, signParams, secret: Uint8Array.from(secret.match(/../g).map(h => parseInt(h, 16))) };
  return KEYS;
}
async function hmac(...parts) {
  const k = await keys();
  const key = await crypto.subtle.importKey('raw', k.secret, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']);
  return new Uint8Array(await crypto.subtle.sign('HMAC', key, enc.encode(parts.join('\x1f'))));
}
async function signToken(claims) {
  const k = await keys();
  const head = b64u(enc.encode(JSON.stringify({ alg: k.alg, typ: 'VLT', kid: k.kid }))), body = b64u(enc.encode(stable(claims)));
  const sig = new Uint8Array(await crypto.subtle.sign(k.signParams, k.priv, enc.encode(`${head}.${body}`)));
  return `${head}.${body}.${b64u(sig)}`;
}
async function openToken(token) {
  const k = await keys();
  try {
    const [head, body, sig] = token.split('.');
    if (JSON.parse(new TextDecoder().decode(b64uBytes(head))).alg !== k.alg) throw new Error('unsupported algorithm');
    if (!(await crypto.subtle.verify(k.signParams, k.pub, b64uBytes(sig), enc.encode(`${head}.${body}`)))) throw new Error('bad signature');
    const claims = JSON.parse(new TextDecoder().decode(b64uBytes(body)));
    if (claims.exp <= now()) return { error: 'token expired' };
    return { claims };
  } catch (e) { return { error: 'invalid token signature or format' }; }
}

// ------------------------------------------------------------------------------------------------ persistent state + audit chain
const GENESIS = '0'.repeat(64);
const S = (() => {
  let st = null;
  try { st = JSON.parse(localStorage.getItem('vl.state') || 'null'); } catch (e) { /* ignore */ }
  return st || { users: {}, sessions: {}, used: {}, events: [] };
})();
const save = () => { try { localStorage.setItem('vl.state', JSON.stringify(S)); } catch (e) { /* memory only */ } };
async function log(type, sid, uid, data = {}) {
  const prev = S.events.length ? S.events[S.events.length - 1].hash : GENESIS;
  const ts = Math.round(now() * 1000) / 1000, payload = stable(data);
  const h = hex(await sha256([prev, String(ts), type, sid || '', uid || '', payload].join('|')));
  S.events.push({ seq: S.events.length + 1, ts, type, session_id: sid || null, user_id: uid || null, data, prev_hash: prev, hash: h });
  if (S.events.length > 500) S.events.splice(0, S.events.length - 500);           // keep the demo log small (chain restarts from the oldest kept event)
  save();
}
export async function verifyChain() {
  let prev = S.events.length ? S.events[0].prev_hash : GENESIS;
  for (const e of S.events) {
    const h = hex(await sha256([prev, String(e.ts), e.type, e.session_id || '', e.user_id || '', stable(e.data)].join('|')));
    if (e.prev_hash !== prev || e.hash !== h) return { valid: false, events: S.events.length, broken_at: e.seq };
    prev = h;
  }
  return { valid: true, events: S.events.length, head: prev };
}

// ------------------------------------------------------------------------------------------------ challenge
async function derive(txnH, sid, nonce, pass) {
  const blocks = [];
  for (let i = 0; i < 3; i++) blocks.push(await hmac('challenge', txnH, sid, nonce, String(pass), String(i)));
  const stream = new Uint8Array(96); blocks.forEach((b, i) => stream.set(b, i * 32));
  const words = []; let i = 0;
  while (words.length < 3) { const w = WORDS[((stream[i] << 8) | stream[i + 1]) % WORDS.length]; if (!words.includes(w)) words.push(w); i += 2; }
  const digits = [0, 1, 2, 3].map(j => stream[40 + j] % 10).join('');
  return { words, digits };
}
const phrase = (w, d) => w.join(' ') + ', ' + d.split('').join(' ');

function tokens(text) {
  const words = [], digits = [];
  for (const t of text.toLowerCase().replace(/-/g, ' ').match(/[a-z]+|\d+/g) || []) {
    if (/^\d+$/.test(t)) digits.push(...t); else if (t in DIGIT_WORDS) digits.push(DIGIT_WORDS[t]); else words.push(t);
  }
  return { words, digits: digits.join('') };
}
function longest(a, b, alo, ahi, blo, bhi) {              // difflib.SequenceMatcher.find_longest_match without junk
  let bi = alo, bj = blo, bk = 0; const prev = new Array(bhi - blo + 1).fill(0);
  for (let i = alo; i < ahi; i++) {
    let last = 0; const row = new Array(bhi - blo + 1).fill(0);
    for (let j = blo; j < bhi; j++) {
      if (a[i] === b[j]) { const k = (prev[j - blo] || 0) + 1; row[j - blo + 1] = k; if (k > bk) { bi = i - k + 1; bj = j - k + 1; bk = k; } }
    }
    for (let j = 0; j <= bhi - blo; j++) prev[j] = row[j];
    void last;
  }
  return [bi, bj, bk];
}
function matchedTotal(a, b, alo = 0, ahi = a.length, blo = 0, bhi = b.length) {
  if (alo >= ahi || blo >= bhi) return 0;
  const [i, j, k] = longest(a, b, alo, ahi, blo, bhi);
  if (!k) return 0;
  return k + matchedTotal(a, b, alo, i, blo, j) + matchedTotal(a, b, i + k, ahi, j + k, bhi);
}
const ratio = (a, b) => (a.length + b.length ? 2 * matchedTotal(a, b) / (a.length + b.length) : 1);
export function matchChallenge(words, digits, transcript, minMatch = CFG.challengeMinMatch) {
  const heard = tokens(transcript); const pool = [...heard.words]; let hits = 0;
  for (const w of words) {
    let best = null, br = -1;
    for (const h of pool) { const r = ratio(w, h); if (r > br) { br = r; best = h; } }
    if (best !== null && br >= 0.8) { hits++; pool.splice(pool.indexOf(best), 1); }
  }
  const dig = matchedTotal(digits, heard.digits);
  const score = (hits + dig) / (words.length + digits.length);
  return { passed: score >= minMatch - 1e-9, score: Math.round(score * 1000) / 1000, words_hit: hits, digits_hit: dig, heard: transcript.trim() };
}

// ------------------------------------------------------------------------------------------------ acoustic analysis (heuristics)
function fft(re, im) {                                     // in-place radix-2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
      }
    }
  }
}
export function acousticAnalysis(y) {
  const n = 400, nf = Math.floor(y.length / n), rms = new Float64Array(nf), db = new Float64Array(nf);
  for (let f = 0; f < nf; f++) { let s = 0; for (let i = 0; i < n; i++) { const v = y[f * n + i]; s += v * v; } rms[f] = Math.sqrt(s / n + 1e-12); db[f] = 20 * Math.log10(rms[f] + 1e-9); }
  const floor = pctl(db, 10), top = pctl(db, 95), dyn = top - floor;
  let silent = 0; for (const r of rms) if (r < 1e-5) silent++;
  let clip = 0; for (let i = 0; i < y.length; i++) if (Math.abs(y[i]) >= 0.999) clip++;
  const N = 512, bins = N / 2 + 1, energy = new Float64Array(bins), re = new Float64Array(N), im = new Float64Array(N);
  for (let f = 0; f < nf; f += 2) {
    re.fill(0); im.fill(0);
    for (let i = 0; i < n; i++) re[i] = y[f * n + i] * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
    fft(re, im);
    for (let k = 0; k < bins; k++) energy[k] += re[k] * re[k] + im[k] * im[k];
  }
  let tot = 1e-12; for (const e of energy) tot += e;
  let high = 0, cum = 0, roll = 8000;
  for (let k = 0; k < bins; k++) { const hz = k * SR / N; if (hz >= 4000) high += energy[k]; cum += energy[k]; if (roll === 8000 && cum / tot >= 0.95) roll = hz; }
  high /= tot;
  const flags = []; let score = 0;
  const sil = silent / Math.max(1, nf), clp = clip / Math.max(1, y.length);
  if (sil > 0.05) { flags.push('digital silence (exact zeros) - typical of synthesised or spliced audio'); score += 0.35; }
  if (floor < -80) { flags.push('unnaturally clean noise floor'); score += 0.25; }
  if (roll < 3500 && high < 0.002) { flags.push('strongly band-limited (phone / loudspeaker / re-recording)'); score += 0.15; }
  if (floor > -40 && dyn < 20) { flags.push('very noisy, low dynamic range (possible loudspeaker re-recording, heuristic)'); score += 0.20; }
  if (clp > 0.01) { flags.push('clipping'); score += 0.10; }
  return { anomaly_score: Math.round(Math.min(score, 1) * 100) / 100, flags, noise_floor_db: Math.round(floor * 10) / 10, dynamic_range_db: Math.round(dyn * 10) / 10,
           rolloff_hz: Math.round(roll), highband_ratio: Math.round(high * 1e4) / 1e4, digital_silence: Math.round(sil * 1e3) / 1e3, clipping: Math.round(clp * 1e4) / 1e4 };
}

// ------------------------------------------------------------------------------------------------ models
const MODELS = { detector: null, speaker: null, asr: null };
const status = { detector: { got: 0, total: 0, state: 'idle' }, speaker: { got: 0, total: 0, state: 'idle' }, asr: { got: 0, total: 0, state: 'idle' } };
let statusCb = () => {};
export const onStatus = cb => { statusCb = cb; };
const setStatus = (k, patch) => { Object.assign(status[k], patch); statusCb(status); };
const base = new URL('../', import.meta.url).href.replace(/\/$/, '');          // site root
const vlBase = new URL('./', import.meta.url).href;

async function loadDetector() {
  setStatus('detector', { state: 'loading' });
  MODELS.detector = await Detector.load((g, t) => setStatus('detector', { got: g, total: t }), base);
  setStatus('detector', { state: 'ready' });
}
async function loadSpeaker(manifest) {
  setStatus('speaker', { state: 'loading', total: manifest.ecapa.bytes });
  const url = vlBase + manifest.ecapa.file;
  const bytes = await fetchRobust(url, { expected: manifest.ecapa.bytes, onProgress: (g, t) => setStatus('speaker', { got: g, total: t }) });
  MODELS.speaker = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  setStatus('speaker', { state: 'ready' });
}
async function loadAsr(manifest) {
  const files = manifest.whisper;                                                // { 'onnx/encoder_model_quantized.onnx': bytes, ... }
  const total = Object.values(files).reduce((a, b) => a + b, 0), got = {};
  setStatus('asr', { state: 'loading', total });
  const tf = await import('./lib/transformers.min.js');
  const env = tf.env;
  env.allowRemoteModels = false; env.allowLocalModels = true; env.localModelPath = vlBase; env.useBrowserCache = false;
  env.backends.onnx.wasm.wasmPaths = vlBase + 'lib/'; env.backends.onnx.wasm.numThreads = 1;
  env.useCustomCache = true;
  env.customCache = {                                                            // route the two big model files through the robust downloader
    async match(req) {
      const url = new URL(typeof req === 'string' ? req : req.url, location.href).href;
      const rel = Object.keys(files).find(f => url.endsWith('/whisper-tiny.en/' + f));
      if (!rel) return undefined;
      const bytes = await fetchRobust(url, { expected: files[rel], key: url + '?v=' + manifest.whisper_version,
        onProgress: g => { got[rel] = g; setStatus('asr', { got: Object.values(got).reduce((a, b) => a + b, 0) }); } });
      return new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } });
    },
    async put() { /* kept by fetchRobust in Cache Storage */ },
  };
  MODELS.asr = await tf.pipeline('automatic-speech-recognition', 'whisper-tiny.en', { dtype: 'q8', device: 'wasm' });
  setStatus('asr', { got: total, state: 'ready' });
}
// What is known about the authenticity limit on AI voices the detector has never seen (scripts/conformal_study.py). conformal.json holds
// `audit` (measured behaviour of the default limits) and `tiers` (strict domain-robust conformal limits). Optional: without the file the
// result says "heuristic". Default mode = audit (default limits + their audited error); "strict" swaps in the conformal limits.
async function loadGuarantee() {
  try {
    const r = await fetch(vlBase + 'conformal.json', { cache: 'no-cache' });
    if (!r.ok) return;
    const cal = await r.json();
    if (TIERS.every(t => cal.tiers && cal.audit && Number.isFinite(cal.tiers[t].p_ai_limit) && cal.audit[t])) CFG.cal = cal;
  } catch (e) { /* heuristic limits */ }
}
export const setStrict = v => { CFG.strict = !!v; };
const isStrict = () => !!(CFG.cal && CFG.strict);
const aiLimitFor = tier => isStrict() ? CFG.cal.tiers[tier].p_ai_limit : CFG.aiLimit[tier];
const guaranteeFor = tier => {
  if (!CFG.cal) return { source: 'heuristic', note: 'hand-set limit, no formal error guarantee' };
  if (isStrict()) {
    const t = CFG.cal.tiers[tier];
    return { source: 'conformal', alpha: t.alpha, confidence: +(1 - CFG.cal.delta).toFixed(3), generators: CFG.cal.generators_used,
             genuine_rejected_pooled: +t.genuine_rejected_pooled.toFixed(3) };
  }
  const a = CFG.cal.audit[tier];
  return { source: 'audit', generators: a.generators, mean_generator_miss: +a.mean_generator_miss.toFixed(3),
           pct_generators_miss_over_10pct: +a.pct_generators_miss_over_10pct.toFixed(3), genuine_rejected: +a.genuine_rejected.toFixed(3) };
};
let loading = null;
export function loadModels() {
  if (!loading) {
    loading = (async () => {
      const manifest = await (await fetch(vlBase + 'manifest.json', { cache: 'no-cache' })).json();
      await Promise.all([loadDetector(), loadSpeaker(manifest), loadAsr(manifest), loadGuarantee()]);
    })().catch(e => { loading = null; throw e; });
  }
  return loading;
}
export const modelsReady = () => Object.values(MODELS).every(Boolean);

async function embed(wave) {
  const out = await MODELS.speaker.run({ audio: new ort.Tensor('float32', Float32Array.from(wave), [1, wave.length]) });
  return unit(out[Object.keys(out)[0]].data);
}
// Speech-only audio for the speaker layers: leading/trailing silence and room tone carry no voice and wreck the embedding.
const speechWave = w => { const t = trimSilence(w, 35); return t.length >= SR ? t : w; };
// Does every 2 s *speech* segment sound like the same speaker as the whole response? Segments that are mostly pauses or room tone are
// skipped (their embeddings are meaningless: on real recordings with pauses the unfiltered check rejected 71 % of genuine clips).
async function consistency(wave, full) {
  const n = 2 * SR, h = SR;
  if (wave.length < n + h) return null;
  const sims = [];
  for (let s = 0; s + n <= wave.length; s += h) {
    const seg = wave.subarray(s, s + n);
    if (speechFraction(seg) >= 0.6) sims.push(dot(await embed(seg), full));
  }
  if (sims.length < 2) return null;
  return { min: Math.min(...sims), mean: sims.reduce((a, b) => a + b, 0) / sims.length, segments: sims.length };
}
function speakerScore(enrolled, emb) {
  const c = new Float32Array(emb.length);
  for (const e of enrolled) for (let i = 0; i < c.length; i++) c[i] += e[i] / enrolled.length;
  const cu = unit(c);
  return { score: dot(cu, emb), best: Math.max(...enrolled.map(e => dot(unit(e), emb))), n_enrolled: enrolled.length };
}
async function forensics(wave) {
  const r = await MODELS.detector.predictWave(wave);
  return { p_ai: r.prob_ai, verdict: r.verdict, windows: r.windows.filter(w => !w.skipped).map(w => ({ start: w.start, p_ai: w.prob_ai })), acoustic: acousticAnalysis(wave) };
}
const transcribe = async wave => (await MODELS.asr(wave)).text;

// ------------------------------------------------------------------------------------------------ risk
function assess(txn, known, recent, t = now()) {
  const amount = parseFloat(txn.amount);
  let base_ = CFG.amountTiers.filter(x => amount >= x).length, idx = base_; const reasons = [`amount ${txn.amount} ${txn.currency}`];
  if (!known.includes(txn.recipient)) { idx++; reasons.push('recipient not previously authorised'); }
  if (recent >= CFG.velocityLimit) { idx++; reasons.push(`${recent} requests in the last hour`); }
  const hour = new Date(t * 1000).getHours();
  if (base_ >= 1 && !(hour >= CFG.offHours[0] && hour < CFG.offHours[1])) { idx++; reasons.push('outside normal hours'); }
  const tier = TIERS[Math.min(idx, 3)];
  return { tier, base_tier: TIERS[Math.min(base_, 3)], risk_score: TIER_RISK[tier], reasons };
}
function judge(tier, challenge, forensic, speaker, cons) {
  const pAi = forensic.p_ai, anomaly = forensic.acoustic.anomaly_score;
  // acoustic anomalies only ever tighten the limit, so a strict (calibrated) guarantee stays valid (it becomes more conservative)
  const aiLimit = isStrict() ? aiLimitFor(tier) * (1 - 0.5 * anomaly) : Math.max(0.05, CFG.aiLimit[tier] - 0.10 * anomaly), spMin = CFG.speakerMin[tier], sp = speaker.score;
  const checks = {
    challenge: { ok: !!challenge.passed, score: challenge.score, heard: challenge.heard },
    authenticity: { ok: pAi < aiLimit, p_ai: +pAi.toFixed(3), limit: +aiLimit.toPrecision(3), verdict: forensic.verdict, guarantee: guaranteeFor(tier) },
    speaker: { ok: sp >= spMin, score: +sp.toFixed(3), min: spMin },
    consistency: { ok: cons === null ? true : cons.min >= CFG.consistencyMin, min: cons === null ? null : +cons.min.toFixed(3), threshold: CFG.consistencyMin },
  };
  const severe = [];
  if (pAi >= CFG.aiSevere) severe.push('audio is almost certainly synthetic');
  if (sp < CFG.speakerSevere) severe.push('voice does not match the enrolled speaker at all');
  const failed = Object.keys(checks).filter(k => !checks[k].ok);
  const ident = Math.min(1, Math.max(0, (sp - 0.10) / 0.60));
  const trust = Math.pow(Math.max(0, 1 - pAi) * ident * Math.max(challenge.score, 0), 1 / 3) * (1 - 0.5 * anomaly);
  return { checks, failed, severe, all_ok: !failed.length, trust_score: +trust.toFixed(3), acoustic: forensic.acoustic };
}

// ------------------------------------------------------------------------------------------------ service (same API shape as the Python server)
function checkAudio(wave) {
  const dur = wave.length / SR;
  for (let i = 0; i < wave.length; i += 997) if (!Number.isFinite(wave[i])) throw new VLError('audio contains invalid samples');
  if (dur < CFG.minAudio) throw new VLError(`audio too short (${dur.toFixed(1)}s); at least ${CFG.minAudio.toFixed(1)}s of speech is needed`);
  if (dur > CFG.maxAudio) throw new VLError(`audio too long (${dur.toFixed(1)}s); maximum is ${CFG.maxAudio}s`);
  return wave;
}
const audioId = async w => hex(await sha256(new Uint8Array(w.buffer, w.byteOffset, Math.min(w.byteLength, 64000)))).slice(0, 16);
const enrolled = u => (S.users[u] ? S.users[u].voiceprints.length : 0);
export const enrollmentStatus = u => ({ user_id: u, samples: enrolled(u), required: CFG.minEnroll, enrolled: enrolled(u) >= CFG.minEnroll });

export async function enroll(userId, wave) {
  userId = userId.trim();
  if (!userId || userId.length > 64) throw new VLError('invalid user id');
  wave = checkAudio(wave);
  let f;
  try { f = await forensics(wave); } catch (e) { if (e instanceof AudioError) throw new VLError(e.message); throw e; }
  if (f.p_ai >= CFG.aiLimit.low) { await log('enroll_rejected', null, userId, { reason: 'synthetic', p_ai: +f.p_ai.toFixed(3) }); throw new VLError('this sample looks synthetic and cannot be used as an enrolment sample', 422); }
  const emb = await embed(speechWave(wave));
  S.users[userId] = S.users[userId] || { voiceprints: [], known: [] };
  S.users[userId].voiceprints.push(f32ToB64(emb)); save();
  await log('enroll', null, userId, { samples: S.users[userId].voiceprints.length, audio: await audioId(wave), p_ai: +f.p_ai.toFixed(3) });
  return enrollmentStatus(userId);
}
export async function resetEnrollment(userId) { if (S.users[userId]) { S.users[userId].voiceprints = []; save(); } await log('enroll_reset', null, userId); return enrollmentStatus(userId); }

async function publicChallenge(sess) {
  const { words, digits } = sess.test_mode ? TEST_PHRASE : await derive(sess.txn_hash, sess.id, sess.challenge_nonce, sess.passes);
  return { challenge_id: sess.challenge_id, phrase: phrase(words, digits), words, digits, expires_at: sess.challenge_expires,
           ttl_s: sess.test_mode ? CFG.testChallengeTtl : CFG.challengeTtl, pass: sess.passes + 1, passes_required: sess.required_passes, test_mode: !!sess.test_mode };
}
async function issueChallenge(sess) {
  Object.assign(sess, { challenge_id: randHex(6), challenge_nonce: randHex(16), challenge_used: false,
    challenge_expires: now() + (sess.test_mode ? CFG.testChallengeTtl : CFG.challengeTtl), status: 'awaiting_response' });
  save();
  return publicChallenge(sess);
}

export async function startTransaction(t) {
  const canon = canonicalTxn(t), user = canon.user_id;
  if (!enrollmentStatus(user).enrolled) throw new VLError('user is not enrolled (voiceprint missing)', 409);
  const tm = now(), recent = Object.values(S.sessions).filter(s => s.user_id === user && s.created >= tm - CFG.velocityWindow).length;
  const a = assess(canon, S.users[user].known, recent, tm), h = await txnHash(canon), sid = 'vl_' + randHex(9);
  const sess = { id: sid, user_id: user, txn: canon, txn_hash: h, tier: a.tier, required_passes: CFG.requiredPasses[a.tier], passes: 0, attempts: 0,
                 status: 'awaiting_response', created: tm, expires: tm + CFG.sessionTtl, challenge_hashes: [], scores: [], test_mode: !!t.test_mode };
  S.sessions[sid] = sess;
  const c = await issueChallenge(sess);
  await log('session_started', sid, user, { tier: a.tier, reasons: a.reasons, txn_hash: h, amount: canon.amount, currency: canon.currency, test_mode: !!t.test_mode });
  return { session_id: sid, tier: a.tier, risk: a, transaction: canon, txn_hash: h, challenge: c, test_mode: !!t.test_mode,
           thresholds: { ai_limit: aiLimitFor(a.tier), guarantee: guaranteeFor(a.tier), speaker_min: CFG.speakerMin[a.tier], passes_required: CFG.requiredPasses[a.tier] } };
}

async function fail(sess, reason, verdict, severe, wave) {
  const sid = sess.id, user = sess.user_id; sess.attempts++;
  const out = { session_id: sid, tier: sess.tier, details: verdict, reason, failed_attempts: sess.attempts };
  if (severe.length || sess.attempts >= CFG.maxAttempts) {
    sess.status = 'blocked'; save();
    const why = severe.length ? severe : [`${sess.attempts} failed attempts`];
    await log('blocked', sid, user, { reason, severe: why, audio: await audioId(wave), checks: verdict.checks || null });
    await log('security_incident', sid, user, { summary: why.join('; '), txn_hash: sess.txn_hash, tier: sess.tier });
    return { ...out, decision: 'blocked', incident: true, message: 'Transaction blocked and a security incident was recorded.' };
  }
  const c = await issueChallenge(sess);
  await log('retry', sid, user, { reason, audio: await audioId(wave), checks: verdict.checks || null });
  return { ...out, decision: 'retry', challenge: c, message: 'Verification failed. Please answer the new challenge.' };
}
async function success(sess, verdict, cHash, wave) {
  const sid = sess.id, user = sess.user_id; sess.passes++; sess.challenge_hashes.push(cHash);
  sess.scores.push({ p_ai: verdict.checks.authenticity.p_ai, speaker: verdict.checks.speaker.score, trust: verdict.trust_score });
  const out = { session_id: sid, tier: sess.tier, details: verdict, passes: sess.passes, passes_required: sess.required_passes };
  if (sess.passes < sess.required_passes) {
    const c = await issueChallenge(sess);
    await log('pass', sid, user, { pass_no: sess.passes, audio: await audioId(wave), trust: verdict.trust_score });
    return { ...out, decision: 'step_up', challenge: c, message: `Pass ${sess.passes}/${sess.required_passes} verified. This transaction needs one more independent challenge.` };
  }
  const t = Math.floor(now()), jti = randHex(12), k = await keys();
  const token = await signToken({ iss: 'voicelock-web', sub: user, sid, jti, iat: t, nbf: t, exp: t + CFG.tokenTtl, txn_hash: sess.txn_hash,
    txn: { amount: sess.txn.amount, currency: sess.txn.currency, recipient: sess.txn.recipient }, tier: sess.tier, challenges: sess.challenge_hashes,
    scores: sess.scores, decision: 'verified', mode: sess.test_mode ? 'test (fixed phrase)' : 'live', alg: k.alg });
  sess.status = 'verified'; save();
  await log('verified', sid, user, { token_jti: jti, trust: verdict.trust_score, audio: await audioId(wave), tier: sess.tier, passes: sess.passes, test_mode: !!sess.test_mode });
  return { ...out, decision: 'verified', token, token_expires_at: t + CFG.tokenTtl, message: 'Verified. A single-use authorisation token was issued for this exact transaction.' };
}

export async function respond(sid, wave) {
  const sess = S.sessions[sid];
  if (!sess) throw new VLError('unknown session', 404);
  const user = sess.user_id, tn = now();
  if (['verified', 'blocked', 'expired'].includes(sess.status)) throw new VLError(`session already ${sess.status}`, 409);
  if (tn > sess.expires) { sess.status = 'expired'; save(); await log('session_expired', sid, user); throw new VLError('session expired', 410); }
  wave = checkAudio(wave);
  if (sess.challenge_used) throw new VLError('this challenge was already answered; request a new one', 409);
  sess.challenge_used = true; save();                                              // single use: burned before the audio is looked at
  if (tn > sess.challenge_expires) return fail(sess, 'challenge expired', { expired: true }, [], wave);
  const { words, digits } = sess.test_mode ? TEST_PHRASE : await derive(sess.txn_hash, sess.id, sess.challenge_nonce, sess.passes);
  const t0 = performance.now();
  let heard, forensic, emb, spk, cons;
  try {
    heard = await transcribe(wave);
    forensic = await forensics(wave);
    const speech = speechWave(wave);
    emb = await embed(speech);
    spk = speakerScore(S.users[user].voiceprints.map(b64ToF32), emb);
    cons = await consistency(speech, emb);
  } catch (e) {
    if (e instanceof AudioError) return fail(sess, `unusable audio: ${e.message}`, { failed: ['audio'], unusable: e.message }, [], wave);
    throw e;
  }
  const latency = +((performance.now() - t0) / 1000).toFixed(2);
  const c = matchChallenge(words, digits, heard);
  const verdict = judge(sess.tier, c, forensic, spk, cons);
  verdict.latency_s = latency; verdict.speaker = { ...spk, consistency: cons }; verdict.authenticity = { p_ai: forensic.p_ai, verdict: forensic.verdict, windows: forensic.windows };
  if (!verdict.all_ok) return fail(sess, 'checks failed: ' + verdict.failed.join(', '), verdict, verdict.severe, wave);
  return success(sess, verdict, hex(await sha256(`${words.join(' ')}|${digits}|${sess.challenge_nonce}`)), wave);
}

export async function verifyToken(token, txn, consume = true) {
  const r = await openToken(token);
  if (r.error) { await log('token_rejected', null, null, { reason: r.error }); return { valid: false, reason: r.error }; }
  const claims = r.claims;
  let h; try { h = await txnHash(txn); } catch (e) { return { valid: false, reason: 'invalid transaction' }; }
  if (h !== claims.txn_hash) { await log('token_rejected', claims.sid, claims.sub, { reason: 'transaction mismatch' }); return { valid: false, reason: 'token was issued for a different transaction' }; }
  if (S.used[claims.jti]) { await log('token_rejected', claims.sid, claims.sub, { reason: 'replay' }); return { valid: false, reason: 'token already used' }; }
  if (consume) {
    S.used[claims.jti] = now();
    const u = S.users[claims.sub]; const rcp = S.sessions[claims.sid] && S.sessions[claims.sid].txn.recipient;
    if (u && rcp && !u.known.includes(rcp)) u.known.push(rcp);
    save(); await log('token_consumed', claims.sid, claims.sub, { jti: claims.jti });
  }
  return { valid: true, claims, consumed: consume };
}
export async function demoTransfer(token, txn) {
  const r = await verifyToken(token, txn, true);
  if (!r.valid) { await log('transfer_denied', null, txn.user_id, { reason: r.reason }); return { status: 'denied', reason: r.reason }; }
  const c = canonicalTxn(txn);
  await log('transfer_executed', r.claims.sid, c.user_id, { amount: c.amount, currency: c.currency, recipient: c.recipient });
  return { status: 'executed', amount: c.amount, currency: c.currency, recipient: txn.recipient, authorised_by: r.claims.sid };
}
export const events = (limit = 60) => ({ events: [...S.events].reverse().slice(0, limit) });
export async function resetAll() { S.users = {}; S.sessions = {}; S.used = {}; S.events = []; save(); }
export { decodeToMono16k };
