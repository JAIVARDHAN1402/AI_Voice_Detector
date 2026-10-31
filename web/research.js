// Renders the conformal study (research/results.json, written by scripts/export_research_web.py) - no result numbers are hard-coded here.
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pc = (v, d = 0) => (v * 100).toFixed(d) + ' %';
const METHODS = { fixed_eer: ['fixed threshold (before)', '#8b95a5'], naive: ['naive conformal', '#d9a21b'], domain_robust: ['domain-robust (ours)', '#3b5bdb'] };
const COLORS = ['#8b95a5', '#3b5bdb', '#2f855a', '#d9a21b', '#c53030'];

function lineChart(svg, xs, series, { ymax = 100, ylabel, xlabel, target }) {
  const W = 420, H = 260, L = 44, R = 12, T = 12, B = 44, pw = W - L - R, ph = H - T - B;
  const x = i => L + (xs.length === 1 ? pw / 2 : pw * i / (xs.length - 1)), y = v => T + ph * (1 - v / ymax);
  let s = '';
  for (let t = 0; t <= 4; t++) { const v = ymax * t / 4; s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${Math.round(v)}</text>`; }
  xs.forEach((lab, i) => { s += `<text x="${x(i)}" y="${H - B + 16}" text-anchor="middle">${esc(lab)}</text>`; });
  s += `<text x="${L + pw / 2}" y="${H - 6}" text-anchor="middle">${esc(xlabel)}</text><text transform="translate(12 ${T + ph / 2}) rotate(-90)" text-anchor="middle">${esc(ylabel)}</text>`;
  if (target !== undefined) s += `<line x1="${L}" x2="${W - R}" y1="${y(target)}" y2="${y(target)}" stroke="currentColor" stroke-dasharray="2 3" opacity=".5"/>`;
  series.forEach(({ color, vals, lo, hi }) => {
    if (lo && hi) s += `<path d="M${vals.map((_, i) => `${x(i)},${y(hi[i])}`).join('L')}L${vals.map((_, i) => `${x(vals.length - 1 - i)},${y(lo[vals.length - 1 - i])}`).join('L')}Z" fill="${color}" opacity=".12"/>`;
    s += `<polyline fill="none" stroke="${color}" stroke-width="2" points="${vals.map((v, i) => `${x(i)},${y(v)}`).join(' ')}"/>` +
      vals.map((v, i) => `<circle cx="${x(i)}" cy="${y(v)}" r="3.2" fill="${color}"/>`).join('');
  });
  svg.innerHTML = s;
}
const legend = (el, items) => { el.innerHTML = items.map(([n, c]) => `<span><i style="background:${c}"></i>${esc(n)}</span>`).join(''); };
const at = (arr, a) => arr.find(r => Math.abs(r.alpha - a) < 1e-9);

async function main() {
  let D;
  try { D = await (await fetch('research/results.json', { cache: 'no-cache' })).json(); } catch (e) { $('coverText').textContent = 'Results are not available.'; return; }
  const xs = D.alphas.map(a => (a * 100).toFixed(0) + ' %'), conf = (1 - D.delta) * 100;
  const dr = D.coverage.domain_robust, nv = D.coverage.naive, fx = D.coverage.fixed_eer, a0 = 0.10;

  $('kpis').innerHTML = [
    [pc(at(dr, a0).corpora), `held-out corpora within a ${pc(a0)} target (domain-robust)`],
    [pc(at(fx, a0).corpora), `same, fixed threshold`],
    [pc(at(dr, a0).frr), `genuine speech rejected by the domain-robust limit`],
    [String(at(dr, a0).units), `held-out generators scored`],
  ].map(([b, t]) => `<div class="kpi"><b>${b}</b><span>${t}</span></div>`).join('');

  const per = key => Object.entries(METHODS).map(([m, [, c]]) => ({ color: c, vals: D.coverage[m].map(r => r[key] * 100) }));
  lineChart($('chCorpora'), xs, per('corpora'), { ymax: 100, ylabel: 'corpora meeting target (%)', xlabel: 'allowed AI-accepted rate', target: conf });
  lineChart($('chGen'), xs, Object.entries(METHODS).map(([m, [, c]]) => ({ color: c, vals: D.coverage[m].map(r => r.meet * 100), lo: D.coverage[m].map(r => r.lo * 100), hi: D.coverage[m].map(r => r.hi * 100) })),
    { ymax: 100, ylabel: 'generators meeting target (%)', xlabel: 'allowed AI-accepted rate', target: conf });
  lineChart($('chPrice'), xs, per('frr'), { ymax: 100, ylabel: 'genuine speech rejected (%)', xlabel: 'allowed AI-accepted rate' });
  legend($('legend'), Object.values(METHODS));
  $('coverText').textContent =
    `At a ${pc(a0)} target, a fixed threshold keeps ${pc(at(fx, a0).corpora)} of held-out corpora within it (mean per-generator miss ${pc(at(fx, a0).mean_miss)}); naive conformal ${pc(at(nv, a0).corpora)}; ` +
    `domain-robust calibration ${pc(at(dr, a0).corpora)} (nominal ${conf.toFixed(0)} %). Per generator the domain-robust figure is ${pc(at(dr, a0).meet)}: generators cluster by corpus, so a hard generator inside a corpus that passes can still exceed the target. ` +
    `The price is ${pc(at(dr, a0).frr)} genuine speech rejected (fixed: ${pc(at(fx, a0).frr)}, naive: ${pc(at(nv, a0).frr)}). ` +
    `Sanity check: ordinary in-domain conformal calibration meets its target (${pc(D.in_domain_check['0.1'], 1)} observed at ${pc(a0)}).`;

  const aud = Object.entries(D.audit);
  $('auditTable').innerHTML = '<tr><th>tier</th><th>default P(AI) limit</th><th>AI accepted, mean over unseen generators</th><th>generators above 10 % accepted</th><th>genuine rejected</th></tr>' +
    aud.map(([t, v]) => `<tr><td>${t}</td><td>${pc(v.p_ai_limit)}</td><td>${pc(v.mean_generator_miss)}</td><td>${pc(v.pct_generators_miss_over_10pct)}</td><td>${pc(v.genuine_rejected)}</td></tr>`).join('');
  $('auditNote').textContent = `Measured leave-one-corpus-out on direct-capture audio over ${aud[0][1].generators} generators. The shipped detector is trained on every corpus, so a truly new generator may do better or worse; these limits carry no formal guarantee, which is why VoiceLock shows this audit next to each tier.`;
  $('tierTable').innerHTML = '<tr><th>tier</th><th>target &alpha;</th><th>P(AI) limit (domain-robust)</th><th>genuine rejected</th></tr>' +
    Object.entries(D.deployment.tiers).map(([t, v]) => `<tr><td>${t}</td><td>${pc(v.alpha)}</td><td>${pc(v.p_ai_limit, 3)}</td><td>${pc(v.genuine_rejected_pooled)}</td></tr>`).join('');

  const rows = [['direct capture (clean + codec/noise/reverb)', D.coverage]].concat(Object.entries(D.other_scopes).map(([k, v]) => [k === 'all' ? 'all conditions incl. replay' : 'loudspeaker &rarr; microphone replay only', v]));
  $('scopeTable').innerHTML = `<tr><th>scope (&alpha; = ${pc(a0)})</th><th>fixed: corpora ok</th><th>naive: corpora ok</th><th>domain-robust: corpora ok</th><th>domain-robust: genuine rejected</th></tr>` +
    rows.map(([n, c]) => `<tr><td>${n}</td><td>${pc(at(c.fixed_eer, a0).corpora)}</td><td>${pc(at(c.naive, a0).corpora)}</td><td>${pc(at(c.domain_robust, a0).corpora)}</td><td>${pc(at(c.domain_robust, a0).frr)}</td></tr>`).join('');
  $('scopeNote').textContent = 'On simulated loudspeaker-to-microphone replay the detector is close to chance, so a bound there is only reached by rejecting almost all genuine speech. VoiceLock relies on its other layers (live challenge, speaker, consistency) for replay.';

  const sp = D.splice, names = Object.keys(sp.detection), ks = Object.keys(sp.detection[names[0]]);
  lineChart($('chSplice'), ks.map((_, i) => `${i + 1}/${sp.M}`), names.map((n, i) => ({ color: COLORS[(i + 0) % COLORS.length], vals: ks.map(k => sp.detection[n][k] * 100) })),
    { ymax: 100, ylabel: `detected at ${pc(sp.false_alarm_on_pure_real)} false alarms (%)`, xlabel: 'share of the recording that is AI' });
  legend($('spliceLegend'), names.map((n, i) => [n, COLORS[i % COLORS.length]]));
  $('spliceText').textContent = `Window-level simulation (${sp.M} windows per recording, direct-capture audio). With one AI window in ${sp.M} the deployed mean rule detects ${pc(sp.detection[names[0]][ks[0]])} and no simple aggregation rule does much better, ` +
    `so a short cloned segment inside real speech is a genuine weakness. The seam between real and AI audio is not modelled.`;

  $('limits').innerHTML = [
    `Leave-one-corpus-out is a hard test: the shipped detector sees every corpus, so these numbers describe generalisation to a new corpus, not the accuracy on familiar ones.`,
    `The domain-robust guarantee is approximate and holds at corpus level (${D.domains.length} corpora); per generator it is lower. It assumes a new corpus resembles the population of calibration corpora.`,
    `Only the AI-accepted rate is bounded. The strict limits reject a large share of genuine speech, so they are an opt-in, not the default.`,
    `Replay is simulated, splicing is simulated at window level, and the official ASVspoof / In-the-Wild protocols, external baselines (AASIST, RawNet2) and an adaptive attacker were not evaluated.`,
  ].map(t => `<li>${t}</li>`).join('');
}
main();
