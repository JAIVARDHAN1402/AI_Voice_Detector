import { Detector, AudioError, decodeToMono16k } from './detector.js';

const $ = s => document.querySelector(s);
const MAX_MB = 25;
let detector = null, busy = false;

// ---------- model loading ----------
async function boot() {
  try {
    detector = await Detector.load((got, total) => {
      const mb = x => (x / 1048576).toFixed(0);
      $('#load-bar').value = total ? (got / total) * 100 : 0;
      $('#load-hint').textContent = mb(got) + ' / ' + (total ? mb(total) : '?') + ' MB';
    });
    window.__detector = detector;       // handy for debugging in the console
    $('#loading').hidden = true; $('#app').hidden = false;
  } catch (e) {
    console.error(e);
    $('#load-text').textContent = 'Could not load the detection model: ' + e.message;
    $('#load-text').className = 'err'; $('#load-bar').hidden = true;
  }
}
boot();

// ---------- tabs ----------
function showTab(name) {
  for (const t of ['upload', 'mic']) {
    $('#tab-' + t).setAttribute('aria-selected', String(t === name));
    $('#pane-' + t).hidden = t !== name;
  }
}
$('#tab-upload').onclick = () => showTab('upload');
$('#tab-mic').onclick = () => showTab('mic');

// ---------- upload ----------
const drop = $('#drop'), fileInput = $('#file');
drop.onclick = () => fileInput.click();
drop.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); } };
['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });
fileInput.onchange = () => { if (fileInput.files[0]) handleFile(fileInput.files[0]); fileInput.value = ''; };

function handleFile(f) {
  if (f.size > MAX_MB * 1024 * 1024) return showError('File is larger than ' + MAX_MB + ' MB.');
  const p = $('#player-up'); p.src = URL.createObjectURL(f); p.hidden = false;
  analyze(f);
}

// ---------- microphone ----------
let mediaRec = null, chunks = [], stream = null, audioCtx = null, analyser = null, raf = 0, t0 = 0, tick = 0;
$('#rec').onclick = async () => {
  if (mediaRec && mediaRec.state === 'recording') return stopRec();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
  } catch (e) {
    return showError('Microphone access was blocked. Allow it in the browser address bar, or use the upload tab.');
  }
  chunks = [];
  mediaRec = new MediaRecorder(stream);
  mediaRec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  mediaRec.onstop = onRecStop;
  audioCtx = new AudioContext();
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024;
  audioCtx.createMediaStreamSource(stream).connect(analyser);
  mediaRec.start();
  t0 = Date.now();
  $('#rec').textContent = '■ Stop & analyze'; $('#rec').classList.add('on');
  tick = setInterval(() => {
    const s = Math.floor((Date.now() - t0) / 1000);
    $('#timer').textContent = Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
    if (s >= 60) stopRec();
  }, 250);
  draw();
};
function stopRec() { mediaRec.stop(); stream.getTracks().forEach(t => t.stop()); }
function onRecStop() {
  clearInterval(tick); cancelAnimationFrame(raf); audioCtx.close();
  $('#rec').textContent = '● Start recording'; $('#rec').classList.remove('on');
  const blob = new Blob(chunks, { type: mediaRec.mimeType || 'audio/webm' });
  const p = $('#player-mic'); p.src = URL.createObjectURL(blob); p.hidden = false;
  if ((Date.now() - t0) < 2000) return showError('Recording too short. Please record at least a few seconds.');
  analyze(blob);
}
function draw() {
  const c = $('#wave'), g = c.getContext('2d'), buf = new Uint8Array(analyser.fftSize);
  const col = getComputedStyle(document.documentElement).getPropertyValue('--accent');
  (function loop() {
    raf = requestAnimationFrame(loop);
    analyser.getByteTimeDomainData(buf);
    g.clearRect(0, 0, c.width, c.height); g.lineWidth = 2; g.strokeStyle = col; g.beginPath();
    for (let i = 0; i < buf.length; i++) { const x = i / buf.length * c.width, y = buf[i] / 255 * c.height; if (i) g.lineTo(x, y); else g.moveTo(x, y); }
    g.stroke();
  })();
}

// ---------- analysis ----------
function showError(msg) {
  $('#result').hidden = true; const s = $('#status'); s.hidden = false;
  s.textContent = ''; const d = document.createElement('div'); d.className = 'err'; d.textContent = msg; s.append(d);
}
function setStatus(text) { const s = $('#status'); s.hidden = false; s.innerHTML = '<span class="spin"></span>'; s.append(text); }

async function analyze(blob) {
  if (busy) return;
  busy = true; $('#result').hidden = true;
  try {
    setStatus('Reading audio…');
    const y = await decodeToMono16k(await blob.arrayBuffer());
    setStatus('Analyzing…');
    const r = await detector.predictWave(y, (i, n) => setStatus('Analyzing segment ' + i + ' of ' + n + '…'));
    $('#status').hidden = true; render(r);
  } catch (e) {
    console.error(e);
    showError(e instanceof AudioError ? e.message : 'Something went wrong while analysing the audio.');
  } finally { busy = false; }
}
function colorFor(p) { return 'hsl(' + (150 - 135 * p) + ' 60% 45%)'; }
function render(j) {
  const b = $('#badge');
  b.className = 'badge ' + j.verdict;
  b.textContent = { real: 'Real voice', ai: 'AI-generated', unsure: 'Uncertain' }[j.verdict];
  $('#headline').textContent = Math.round(j.prob_ai * 100) + '% probability AI-generated';
  $('#subline').textContent = j.duration.toFixed(1) + ' s analysed · ' + j.n_windows + ' segment' + (j.n_windows > 1 ? 's' : '');
  $('#needle').style.left = (j.prob_ai * 100) + '%';
  const segs = $('#segs'); segs.textContent = '';
  j.windows.forEach(w => {
    const e = document.createElement('span');
    e.style.background = w.skipped ? 'var(--line)' : colorFor(w.prob_ai);
    e.title = w.skipped ? (w.start.toFixed(1) + 's: silence (skipped)') : (w.start.toFixed(1) + 's: ' + Math.round(w.prob_ai * 100) + '% AI');
    segs.append(e);
  });
  const n = $('#notes'); n.textContent = '';
  j.notes.forEach(t => { const li = document.createElement('li'); li.textContent = t; n.append(li); });
  $('#result').hidden = false;
}
