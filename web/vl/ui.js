import * as VL from './engine.js';

const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = v => (v * 100).toFixed(0) + '%';
const user = () => $('user').value.trim();
const PROMPTS = [
  'The quick brown fox jumps over the lazy dog near the old river bridge.',
  'Please confirm that I have reviewed this payment and I am the person making it.',
  'Seven green lanterns glowed softly while the market closed for the evening.',
  'My voice is my password, and nobody else may speak for me today.',
];
let session = null, lastTxn = null, lastToken = null, timerId = null, expiresAt = 0, promptIdx = 0, ready = false;
const hasMic = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder);
if (!hasMic) { $('enrollRec').classList.add('hidden'); $('rec').classList.add('hidden'); }

const wave = async blob => VL.decodeToMono16k(await blob.arrayBuffer());
const fail = e => alert((e && e.message) || String(e));

/* ---------------- models ---------------- */
const NAMES = { detector: 'AI-voice detector (WavLM)', speaker: 'Speaker model (ECAPA-TDNN)', asr: 'Speech model (Whisper tiny.en)' };
const mb = n => (n / 1e6).toFixed(n > 1e7 ? 0 : 1);
function drawModels(st) {
  $('models').innerHTML = Object.keys(NAMES).map(k => {
    const s = st[k], p = s.total ? Math.min(100, s.got / s.total * 100) : 0;
    const label = s.state === 'ready' ? '<span class="ok">ready</span>' : s.state === 'error' ? '<span class="bad">failed</span>' : s.state === 'loading' ? `${mb(s.got)} / ${s.total ? mb(s.total) : '?'} MB` : 'waiting';
    return `<div class="model"><b>${NAMES[k]}<span>${label}</span></b><div class="meter"><div style="width:${s.state === 'ready' ? 100 : p}%"></div></div></div>`;
  }).join('');
}
VL.onStatus(drawModels);
drawModels({ detector: { state: 'idle' }, speaker: { state: 'idle' }, asr: { state: 'idle' } });

async function startLoading() {
  $('loadBtn').classList.add('hidden'); $('modelsChip').textContent = 'loading'; $('modelsChip').className = 'chip';
  try {
    await VL.loadModels();
    ready = true; $('modelsChip').textContent = 'ready'; $('modelsChip').className = 'chip ok'; $('start').disabled = false;
    $('modelsNote').textContent = 'Models are loaded and cached in this browser. Nothing leaves your device.';
  } catch (e) {
    $('modelsChip').textContent = 'failed'; $('modelsChip').className = 'chip bad'; $('loadBtn').classList.remove('hidden');
    $('modelsNote').textContent = 'Loading failed: ' + ((e && e.message) || e) + ' - tap Retry (downloads resume from the cache).';
  }
}
$('loadBtn').addEventListener('click', startLoading);

/* ---------------- recording (optional) ---------------- */
function makeRecorder(button, idleText, onBlob) {
  let rec = null, chunks = [], stream = null;
  button.addEventListener('click', async () => {
    if (rec) { rec.stop(); return; }
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } }); }
    catch (e) { alert('Microphone unavailable (' + e.message + '). Upload an audio file instead.'); return; }
    chunks = []; rec = new MediaRecorder(stream);
    rec.ondataavailable = e => e.data.size && chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach(t => t.stop());
      button.classList.remove('rec'); button.textContent = 'Processing…'; button.disabled = true;
      const blob = new Blob(chunks, { type: rec.mimeType }); rec = null;
      try { await onBlob(blob); } catch (e) { fail(e); }
      button.disabled = false; button.textContent = idleText;
    };
    rec.start(); button.classList.add('rec'); button.textContent = 'Stop recording';
  });
}

/* ---------------- 1. enrolment ---------------- */
function refreshEnrollment() {
  const s = VL.enrollmentStatus(user());
  $('enrollBar').style.width = Math.min(100, s.samples / s.required * 100) + '%';
  $('enrollText').textContent = `${s.samples} / ${s.required} samples` + (s.enrolled ? ' — enrolled' : '');
  $('enrollPrompt').textContent = s.enrolled || !hasMic ? '' : 'Read aloud: “' + PROMPTS[promptIdx % PROMPTS.length] + '”';
}
async function enrollWith(blob) {
  if (!ready) return alert('The models are still loading.');
  try { await VL.enroll(user(), await wave(blob)); promptIdx++; } catch (e) { fail(e); }
  refreshEnrollment(); refreshLog();
}
makeRecorder($('enrollRec'), 'Record sample', enrollWith);
$('enrollFile').addEventListener('change', async e => { for (const f of e.target.files) await enrollWith(f); e.target.value = ''; });
$('enrollReset').addEventListener('click', async () => { await VL.resetEnrollment(user()); promptIdx = 0; refreshEnrollment(); refreshLog(); });
$('user').addEventListener('change', refreshEnrollment);

/* ---------------- 2. transaction ---------------- */
$('start').addEventListener('click', async () => {
  lastTxn = { user_id: user(), amount: $('amount').value.trim(), currency: $('currency').value.trim() || 'INR',
              recipient: $('recipient').value.trim(), reference: $('reference').value.trim() };
  $('start').disabled = true;
  try {
    const body = await VL.startTransaction({ ...lastTxn, test_mode: $('testMode').checked });
    session = body; lastToken = null; $('c-result').classList.add('hidden');
    const r = body.risk;
    $('risk').classList.remove('hidden');
    $('risk').innerHTML = `Risk tier <b class="${r.tier}">${r.tier.toUpperCase()}</b> &middot; needs ${body.thresholds.passes_required} challenge${body.thresholds.passes_required > 1 ? 's' : ''}` +
      ` &middot; P(AI) must stay below ${pct(body.thresholds.ai_limit)} &middot; speaker match &ge; ${body.thresholds.speaker_min}<br><span class="muted">${r.reasons.map(esc).join(' &middot; ')}</span>`;
    showChallenge(body.challenge);
  } catch (e) { fail(e); }
  $('start').disabled = !ready; refreshLog();
});

/* ---------------- 3. challenge ---------------- */
function showChallenge(c) {
  $('c-challenge').classList.remove('hidden');
  $('phrase').textContent = c.phrase;
  $('passChip').textContent = `pass ${c.pass} of ${c.passes_required}`;
  $('testBadge').classList.toggle('hidden', !c.test_mode);
  $('txnLine').textContent = `${lastTxn.currency.toUpperCase()} ${lastTxn.amount} → ${lastTxn.recipient}`;
  expiresAt = c.expires_at * 1000; clearInterval(timerId);
  const tick = () => { const left = Math.max(0, Math.round((expiresAt - Date.now()) / 1000)); $('timer').textContent = left ? `expires in ${left}s` : 'expired — answering now will count as a failed attempt'; };
  tick(); timerId = setInterval(tick, 500);
  $('c-challenge').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
let busy = false;
async function answer(blob) {
  if (!session || busy) return;
  busy = true; $('banner').textContent = '';
  const prev = $('c-result'); prev.classList.remove('hidden'); $('layers').innerHTML = ''; $('tokenBox').classList.add('hidden');
  $('banner').className = 'banner retry'; $('banner').textContent = 'Analysing the audio on your device (speech, speaker and AI-voice checks)…';
  try { showResult(await VL.respond(session.session_id, await wave(blob))); }
  catch (e) { $('c-result').classList.add('hidden'); fail(e); }
  busy = false; refreshLog();
}
makeRecorder($('rec'), 'Record with microphone', answer);
$('respFile').addEventListener('change', e => { if (e.target.files[0]) answer(e.target.files[0]); e.target.value = ''; });

/* ---------------- 4. decision ---------------- */
function layerCard(title, ok, rows, flags) {
  return `<div class="layer"><h4>${title}<span class="${ok ? 'ok' : 'bad'}">${ok ? '✓ pass' : '✗ fail'}</span></h4><dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('')}</dl>${flags && flags.length ? '<ul>' + flags.map(f => `<li>${esc(f)}</li>`).join('') + '</ul>' : ''}</div>`;
}
function showResult(r) {
  $('c-result').classList.remove('hidden');
  const titles = { verified: 'Verified', blocked: 'Blocked — security incident recorded', retry: 'Not verified — try the new challenge', step_up: 'Pass accepted — one more challenge needed' };
  $('banner').className = 'banner ' + r.decision;
  $('banner').innerHTML = `${titles[r.decision] || r.decision}<small>${esc(r.message || '')}${r.reason ? ' (' + esc(r.reason) + ')' : ''}</small>`;
  const d = r.details || {}, c = d.checks; $('layers').innerHTML = '';
  if (c) {
    const a = d.acoustic || {};
    $('layers').innerHTML =
      layerCard('1 &middot; AI-voice forensics', c.authenticity.ok, [['P(AI)', pct(c.authenticity.p_ai)], ['limit', '< ' + pct(c.authenticity.limit)], ['detector', c.authenticity.verdict],
        ['acoustic anomaly', a.anomaly_score ?? 0], ['noise floor', (a.noise_floor_db ?? '') + ' dB']], a.flags) +
      layerCard('2 &middot; Speaker identity', c.speaker.ok && c.consistency.ok, [['match', c.speaker.score], ['required', '≥ ' + c.speaker.min],
        ['voice consistency', c.consistency.min === null ? 'n/a (short)' : c.consistency.min]]) +
      layerCard('3 &middot; Transaction challenge', c.challenge.ok, [['match', pct(c.challenge.score)], ['heard', c.challenge.heard || '(nothing)']]) +
      layerCard('4 &middot; Risk &amp; authorisation', r.decision === 'verified' || r.decision === 'step_up', [['tier', r.tier], ['trust score', d.trust_score],
        ['passes', r.passes !== undefined ? `${r.passes}/${r.passes_required}` : '—'], ['time', (d.latency_s ?? '—') + ' s']], d.severe || []);
  }
  $('tokenBox').classList.toggle('hidden', !r.token);
  if (r.token) {
    lastToken = r.token;
    $('claims').textContent = JSON.stringify(JSON.parse(atob(r.token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/'))), null, 2); $('execOut').textContent = '';
    clearInterval(timerId); $('c-challenge').classList.add('hidden');
  } else if (r.challenge) { showChallenge(r.challenge); }
  else { clearInterval(timerId); $('c-challenge').classList.add('hidden'); }
  $('c-result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
async function execute(txn, label) {
  const b = await VL.demoTransfer(lastToken, txn);
  $('execOut').innerHTML = `<b>${label}:</b> ` + (b.status === 'executed' ? `✓ executed ${esc(b.currency)} ${esc(b.amount)} to ${esc(b.recipient)}` : `✗ denied — ${esc(b.reason)}`);
  refreshLog();
}
$('execute').addEventListener('click', () => execute(lastTxn, 'Execute'));
$('tamper').addEventListener('click', () => execute({ ...lastTxn, amount: String(Number(lastTxn.amount) * 10) }, 'Tampered amount'));
$('replay').addEventListener('click', () => execute(lastTxn, 'Replay'));

/* ---------------- security log ---------------- */
async function refreshLog() {
  const ev = VL.events(40), ch = await VL.verifyChain();
  $('chain').textContent = ch.valid ? `hash chain intact · ${ch.events} events` : `CHAIN BROKEN at event ${ch.broken_at}`;
  $('chain').className = 'chip ' + (ch.valid ? 'ok' : 'bad');
  $('log').querySelector('tbody').innerHTML = ev.events.map(e => {
    const d = Object.entries(e.data).filter(([k]) => !['checks', 'audio'].includes(k)).map(([k, v]) => `${k}=${typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ');
    return `<tr class="e-${esc(e.type)}"><td>${e.seq}</td><td>${new Date(e.ts * 1000).toLocaleTimeString()}</td><td>${esc(e.type)}</td><td>${esc(e.session_id || '')}</td><td class="d">${esc(d)}</td></tr>`;
  }).join('');
}
$('refreshLog').addEventListener('click', refreshLog);
$('resetAll').addEventListener('click', async () => { if (confirm('Delete all enrolments, sessions and log entries stored in this browser?')) { await VL.resetAll(); refreshEnrollment(); refreshLog(); } });

refreshEnrollment(); refreshLog(); startLoading();
window.VL = VL;            // handy for debugging in the console
