'use strict';
const $ = id => document.getElementById(id);
const SR = 16000;
const ENROLL_PROMPTS = [
  'The quick brown fox jumps over the lazy dog near the old river bridge.',
  'Please confirm that I have reviewed this payment and I am the person making it.',
  'Seven green lanterns glowed softly while the market closed for the evening.',
  'My voice is my password, and nobody else may speak for me today.',
];
let session = null, lastTxn = null, lastToken = null, timerId = null, expiresAt = 0, enrollPromptIdx = 0;

async function api(path, opts = {}) {
  const res = await fetch('/v1' + path, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { /* non-JSON error */ }
  if (!res.ok && !(body && (body.status || body.valid === false))) throw new Error((body && (body.error || body.detail)) || res.statusText);
  return { status: res.status, body };
}
const postJson = (path, data) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
const postAudio = (path, wav) => { const f = new FormData(); f.append('audio', wav, 'audio.wav'); return api(path, { method: 'POST', body: f }); };
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const user = () => $('user').value.trim();

/* ---------- recording: MediaRecorder -> 16 kHz mono WAV in the browser ---------- */
async function toWav16k(blob) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
  ctx.close();
  const off = new OfflineAudioContext(1, Math.ceil(buf.duration * SR), SR);
  const src = off.createBufferSource(); src.buffer = buf; src.connect(off.destination); src.start();
  const mono = (await off.startRendering()).getChannelData(0);
  const out = new DataView(new ArrayBuffer(44 + mono.length * 2));
  const w = (o, s) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); out.setUint32(4, 36 + mono.length * 2, true); w(8, 'WAVEfmt '); out.setUint32(16, 16, true); out.setUint16(20, 1, true);
  out.setUint16(22, 1, true); out.setUint32(24, SR, true); out.setUint32(28, SR * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  w(36, 'data'); out.setUint32(40, mono.length * 2, true);
  for (let i = 0; i < mono.length; i++) out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, mono[i])) * 32767, true);
  return new Blob([out], { type: 'audio/wav' });
}

function makeRecorder(button, idleText, onWav) {
  let rec = null, chunks = [], stream = null;
  button.addEventListener('click', async () => {
    if (rec) { rec.stop(); return; }
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
    catch (e) { alert('Microphone unavailable: ' + e.message); return; }
    chunks = []; rec = new MediaRecorder(stream);
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      button.classList.remove('rec'); button.textContent = 'Processing…'; button.disabled = true;
      const blob = new Blob(chunks, { type: rec.mimeType }); rec = null;
      try { await onWav(await toWav16k(blob)); } catch (e) { alert(e.message); }
      button.disabled = false; button.textContent = idleText;
    };
    rec.start(); button.classList.add('rec'); button.textContent = 'Stop recording';
  });
}

/* ---------- 1. enrolment ---------- */
async function refreshEnrollment() {
  try {
    const { body: s } = await api('/users/' + encodeURIComponent(user()) + '/enrollment');
    $('enrollBar').style.width = Math.min(100, s.samples / s.required * 100) + '%';
    $('enrollText').textContent = `${s.samples} / ${s.required} samples` + (s.enrolled ? ' — enrolled' : '');
    $('enrollPrompt').textContent = s.enrolled ? '' : 'Read aloud: “' + ENROLL_PROMPTS[enrollPromptIdx % ENROLL_PROMPTS.length] + '”';
  } catch (e) { $('enrollText').textContent = e.message; }
}
async function enrollWith(wav) {
  try { await postAudio('/users/' + encodeURIComponent(user()) + '/enroll', wav); enrollPromptIdx++; }
  catch (e) { alert(e.message); }
  refreshEnrollment(); refreshLog();
}
makeRecorder($('enrollRec'), 'Record sample', enrollWith);
$('enrollFile').addEventListener('change', e => { if (e.target.files[0]) enrollWith(e.target.files[0]); e.target.value = ''; });
$('enrollReset').addEventListener('click', async () => { await api('/users/' + encodeURIComponent(user()) + '/enrollment', { method: 'DELETE' }); enrollPromptIdx = 0; refreshEnrollment(); refreshLog(); });
$('user').addEventListener('change', refreshEnrollment);

/* ---------- 2. transaction ---------- */
$('start').addEventListener('click', async () => {
  lastTxn = { user_id: user(), amount: $('amount').value.trim(), currency: $('currency').value.trim() || 'INR',
              recipient: $('recipient').value.trim(), reference: $('reference').value.trim() };
  $('start').disabled = true;
  try {
    const { body } = await postJson('/transactions', lastTxn);
    session = body; lastToken = null;
    $('c-result').classList.add('hidden');
    const r = body.risk;
    $('risk').classList.remove('hidden');
    $('risk').innerHTML = `Risk tier <b class="${r.tier}">${r.tier.toUpperCase()}</b> &middot; needs ${body.thresholds.passes_required} challenge${body.thresholds.passes_required > 1 ? 's' : ''}` +
      ` &middot; P(AI) must stay below ${pct(body.thresholds.ai_limit)} &middot; speaker match &ge; ${body.thresholds.speaker_min}<br><span class="muted">${guaranteeText(body.thresholds.guarantee)}</span><br><span class="muted">${r.reasons.map(esc).join(' &middot; ')}</span>`;
    showChallenge(body.challenge);
  } catch (e) { alert(e.message); }
  $('start').disabled = false;
});

/* ---------- 3. challenge ---------- */
function showChallenge(c) {
  $('c-challenge').classList.remove('hidden');
  $('phrase').textContent = c.phrase;
  $('passChip').textContent = `pass ${c.pass} of ${c.passes_required}`;
  $('txnLine').textContent = `${lastTxn.currency.toUpperCase()} ${lastTxn.amount} → ${lastTxn.recipient}`;
  expiresAt = c.expires_at * 1000;
  clearInterval(timerId);
  const tick = () => {
    const left = Math.max(0, Math.round((expiresAt - Date.now()) / 1000));
    $('timer').textContent = left ? `expires in ${left}s` : 'expired — answering now will count as a failed attempt';
  };
  tick(); timerId = setInterval(tick, 500);
  $('c-challenge').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
async function answer(wav) {
  if (!session) return;
  $('rec').disabled = true;
  try {
    const { body } = await postAudio(`/sessions/${session.session_id}/respond`, wav);
    showResult(body);
  } catch (e) { alert(e.message); }
  $('rec').disabled = false; refreshLog();
}
makeRecorder($('rec'), 'Start recording', answer);
$('respFile').addEventListener('change', e => { if (e.target.files[0]) answer(e.target.files[0]); e.target.value = ''; });

/* ---------- 4. decision ---------- */
const pct = v => (v * 100).toFixed(v < 0.001 ? 4 : v < 0.1 ? 1 : 0) + '%';
/* what is known about the authenticity limit on AI voices from generators the detector has never seen */
function guaranteeText(g) {
  if (!g) return '';
  if (g.source === 'conformal') return `strict limit: AI voice from an unseen corpus accepted ≤ ${(g.alpha * 100).toFixed(0)}% (${(g.confidence * 100).toFixed(0)}% confidence); genuine speech rejected ~${(g.genuine_rejected_pooled * 100).toFixed(0)}%`;
  if (g.source === 'audit') return `audited on unseen generators: ~${(g.mean_generator_miss * 100).toFixed(0)}% of AI voices accepted on average, ~${(g.genuine_rejected * 100).toFixed(0)}% of genuine speech rejected (no formal guarantee)`;
  return 'hand-set limit (no formal guarantee)';
}
function layerCard(title, ok, rows, flags) {
  return `<div class="layer"><h4>${title}<span class="${ok ? 'ok' : 'bad'}">${ok ? '✓ pass' : '✗ fail'}</span></h4><dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('')}</dl>${flags && flags.length ? '<ul>' + flags.map(f => `<li>${esc(f)}</li>`).join('') + '</ul>' : ''}</div>`;
}
function showResult(r) {
  $('c-result').classList.remove('hidden');
  const titles = { verified: 'Verified', blocked: 'Blocked — security incident recorded', retry: 'Not verified — try the new challenge', step_up: 'Pass accepted — one more challenge needed' };
  $('banner').className = 'banner ' + r.decision;
  $('banner').innerHTML = `${titles[r.decision] || r.decision}<small>${esc(r.message || '')}${r.reason ? ' (' + esc(r.reason) + ')' : ''}</small>`;
  const d = r.details || {}, c = d.checks;
  const layers = $('layers'); layers.innerHTML = '';
  if (c) {
    const a = d.acoustic || {};
    layers.innerHTML =
      layerCard('1 &middot; AI-voice forensics', c.authenticity.ok, [['P(AI)', pct(c.authenticity.p_ai)], ['limit', '< ' + pct(c.authenticity.limit)], ['guarantee', guaranteeText(c.authenticity.guarantee)], ['detector', c.authenticity.verdict],
        ['acoustic anomaly', a.anomaly_score ?? 0], ['noise floor', (a.noise_floor_db ?? '') + ' dB']], a.flags) +
      layerCard('2 &middot; Speaker identity', c.speaker.ok && c.consistency.ok, [['match', c.speaker.score], ['required', '≥ ' + c.speaker.min],
        ['voice consistency', c.consistency.min === null ? 'n/a (short)' : c.consistency.min]]) +
      layerCard('3 &middot; Transaction challenge', c.challenge.ok, [['match', pct(c.challenge.score)], ['heard', c.challenge.heard || '(nothing)']]) +
      layerCard('4 &middot; Risk &amp; authorisation', r.decision === 'verified' || r.decision === 'step_up', [['tier', r.tier], ['trust score', d.trust_score], ['passes', r.passes !== undefined ? `${r.passes}/${r.passes_required}` : '—'],
        ['latency', (d.latency_s ?? '—') + ' s']], (d.severe || []));
  }
  $('tokenBox').classList.toggle('hidden', !r.token);
  if (r.token) {
    lastToken = r.token;
    const claims = JSON.parse(atob(r.token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/')));
    $('claims').textContent = JSON.stringify(claims, null, 2); $('execOut').textContent = '';
    clearInterval(timerId); $('c-challenge').classList.add('hidden');
  } else if (r.challenge) { showChallenge(r.challenge); }
  else { clearInterval(timerId); $('c-challenge').classList.add('hidden'); }
  $('c-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
async function execute(txn, label) {
  const res = await postJson('/demo/transfer', { token: lastToken, transaction: txn });
  const b = res.body;
  $('execOut').innerHTML = `<b>${label}:</b> ` + (b.status === 'executed' ? `✓ executed ${esc(b.currency)} ${esc(b.amount)} to ${esc(b.recipient)}` : `✗ denied — ${esc(b.reason)}`);
  refreshLog();
}
$('execute').addEventListener('click', () => execute(lastTxn, 'Execute'));
$('tamper').addEventListener('click', () => execute({ ...lastTxn, amount: String(Number(lastTxn.amount) * 10) }, 'Tampered amount'));
$('replay').addEventListener('click', () => execute(lastTxn, 'Replay'));

/* ---------- security log ---------- */
async function refreshLog() {
  try {
    const [{ body: ev }, { body: ch }] = await Promise.all([api('/events?limit=40'), api('/events/verify')]);
    $('chain').textContent = ch.valid ? `hash chain intact · ${ch.events} events` : `CHAIN BROKEN at event ${ch.broken_at}`;
    $('chain').className = 'chip ' + (ch.valid ? 'ok' : 'bad');
    $('log').querySelector('tbody').innerHTML = ev.events.map(e => {
      const d = Object.entries(e.data).filter(([k]) => !['checks', 'audio'].includes(k)).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ');
      return `<tr class="e-${esc(e.type)}"><td>${e.seq}</td><td>${new Date(e.ts * 1000).toLocaleTimeString()}</td><td>${esc(e.type)}</td><td>${esc(e.session_id || '')}</td><td class="d">${esc(d)}</td></tr>`;
    }).join('');
  } catch (e) { /* offline */ }
}
$('refreshLog').addEventListener('click', refreshLog);

(async function init() {
  try { const { body } = await api('/health'); $('status').textContent = 'backend ready · key ' + body.key_id.slice(0, 8); $('status').classList.add('ok'); }
  catch (e) { $('status').textContent = 'backend unreachable'; $('status').classList.add('bad'); }
  refreshEnrollment(); refreshLog();
})();
