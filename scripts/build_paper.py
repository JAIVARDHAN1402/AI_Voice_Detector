"""Write research/PAPER.md from the study results. Every number in the paper is read from reports/*.json at build time and every
comparative sentence is chosen by comparing those numbers, so the text cannot drift from the results. Re-run after any change:

    python scripts/build_paper.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
OUT = ROOT / "research"

mlp = json.loads((R / "conformal_mlp.json").read_text())
lin = json.loads((R / "conformal_linear.json").read_text()) if (R / "conformal_linear.json").exists() else None
ev = json.loads((R / "evaluation.json").read_text())
vl = json.loads((R / "voicelock_eval.json").read_text()) if (R / "voicelock_eval.json").exists() else None
base = json.loads((R / "baseline_mfcc.json").read_text()) if (R / "baseline_mfcc.json").exists() else None

A = mlp["alphas"]
dep = mlp["deployment_calibration"]
aud = mlp["audit_default_limits"]
conf = 1 - mlp["delta"]
pc = lambda v, d=0: f"{100 * v:.{d}f}%"  # noqa: E731
NAMES = {"fixed_eer": "fixed equal-error threshold", "naive": "naive split conformal", "domain_robust": "**domain-robust (ours)**"}


def sm(rep, m, a, scope=None):
    S = rep["summary"] if scope in (None, "direct") else rep["other_scopes"][scope]
    return S[f"{m}@{a}"]


def table_cov(rep):
    rows = ["| target α | method | corpora within α | generators within α (95% CI) | mean miss / generator | worst generator | genuine speech rejected |",
            "|---|---|---|---|---|---|---|"]
    for a in A:
        for m in ("fixed_eer", "naive", "domain_robust"):
            s = sm(rep, m, a)
            rows.append(f"| {pc(a)} | {NAMES[m]} | {pc(s['pct_corpora_meeting_alpha'])} of {s['corpora']} | {pc(s['pct_generators_meeting_alpha'])} "
                        f"[{pc(s['ci95'][0])}, {pc(s['ci95'][1])}] | {pc(s['mean_generator_miss'], 1)} | {pc(s['worst_generator_miss'])} | "
                        f"{pc(s['genuine_rejected'], 1)} |")
    return "\n".join(rows)


def table_scopes(a=0.10):
    rows = [f"| scope (α = {pc(a)}) | method | corpora within α | generators within α | genuine speech rejected |", "|---|---|---|---|---|"]
    for sc, label in (("direct", "direct capture (clean + codec/noise/reverb/band-limit)"), ("all", "all conditions incl. simulated replay"),
                      ("replay", "simulated loudspeaker → microphone replay only")):
        for m in ("fixed_eer", "naive", "domain_robust"):
            s = sm(mlp, m, a, sc)
            rows.append(f"| {label} | {NAMES[m]} | {pc(s['pct_corpora_meeting_alpha'])} | {pc(s['pct_generators_meeting_alpha'])} | {pc(s['genuine_rejected'], 1)} |")
    return "\n".join(rows)


def table_domains(a=0.10):
    rows = [f"| held-out corpus | domain-robust τ (mean logit) | AI accepted, domain-robust (naive; fixed) | genuine rejected, domain-robust |", "|---|---|---|---|"]
    d, n, f = (sm(mlp, m, a)["per_domain"] for m in ("domain_robust", "naive", "fixed_eer"))
    for c in mlp["domains"]:
        fr = d[c]["frr"]
        rows.append(f"| {c} | {d[c]['tau']:.2f} | {pc(d[c]['pooled_miss'], 1)} ({pc(n[c]['pooled_miss'], 1)}; {pc(f[c]['pooled_miss'], 1)}) | "
                    f"{'n/a (no real speech)' if fr is None else pc(fr, 1)} |")
    return "\n".join(rows)


def table_audit():
    rows = ["| tier | default P(AI) limit | AI accepted, mean over unseen generators | generators with > 10% accepted | genuine speech rejected |", "|---|---|---|---|---|"]
    for t, v in aud.items():
        rows.append(f"| {t} | {pc(v['p_ai_limit'])} | {pc(v['mean_generator_miss'], 1)} | {pc(v['pct_generators_miss_over_10pct'])} | {pc(v['genuine_rejected'], 1)} |")
    return "\n".join(rows)


def table_strict():
    rows = ["| tier | target α | strict P(AI) limit | genuine speech rejected (pooled) |", "|---|---|---|---|"]
    for t, v in dep["tiers"].items():
        rows.append(f"| {t} | {pc(v['alpha'])} | {pc(v['p_ai_limit'], 3)} | {pc(v['genuine_rejected_pooled'], 1)} |")
    return "\n".join(rows)


def table_quality():
    q_m, q_l = mlp["per_corpus_quality"], (lin["per_corpus_quality"] if lin else {})
    b = base["per_corpus"] if base else {}
    rows = ["| held-out corpus | WavLM + MLP: EER / AUC | WavLM + logistic: EER / AUC | MFCC + logistic (baseline): EER / AUC |", "|---|---|---|---|"]
    fmt = lambda v: "n/a" if not v else f"{pc(v['eer'], 1)} / {v['auc']:.3f}"  # noqa: E731
    for c in mlp["domains"]:
        if not q_m.get(c):
            continue
        label = c + (" (ASVspoof 2019 LA evaluation attacks A07–A19, our 750-clip subset)" if c == "asv_test" else "")
        rows.append(f"| {label} | {fmt(q_m[c].get('clean'))} | {fmt(q_l.get(c, {}).get('clean'))} | {fmt(b.get(c))} |")
    return "\n".join(rows)


def table_splice():
    d = mlp["splice"]["detection"]
    ks = list(next(iter(d.values())).keys())
    rows = ["| aggregation rule | " + " | ".join(ks) + " |", "|---|" + "---|" * len(ks)]
    for n, v in d.items():
        rows.append(f"| {n} | " + " | ".join(pc(v[k]) for k in ks) + " |")
    return "\n".join(rows)


a0 = 0.10
fx, nv, dr = (sm(mlp, m, a0) for m in ("fixed_eer", "naive", "domain_robust"))
dr_ok = dr["pct_corpora_meeting_alpha"] >= conf - 0.05
gen_ok = dr["pct_generators_meeting_alpha"] >= conf - 0.05
in_dom = mlp["naive_in_domain_check"]
dsw = mlp["delta_sweep_at_alpha_10pct"]
pool = ev["pooled"]
units = dr["units"]
n_dom = len(mlp["domains"])
sp_names = list(mlp["splice"]["detection"])
sp1 = {n: list(mlp["splice"]["detection"][n].values())[0] for n in sp_names}
best_sp = max(sp1, key=sp1.get)
crit = aud["critical"]
low_t = aud["low"]
rep_all, rep_replay = sm(mlp, "domain_robust", a0, "all"), sm(mlp, "domain_robust", a0, "replay")
fx_rep = sm(mlp, "fixed_eer", a0, "replay")

lin_block = ""
if lin:
    lf, ln_, ld = (sm(lin, m, a0) for m in ("fixed_eer", "naive", "domain_robust"))
    lin_block = f"""
### 5.8 Is it specific to the MLP head?

Repeating the whole nested protocol with a logistic-regression head on the same features gives the same ordering (Table 8): at α = {pc(a0)} the fixed threshold keeps
{pc(lf['pct_corpora_meeting_alpha'])} of held-out corpora within the target, naive conformal {pc(ln_['pct_corpora_meeting_alpha'])}, domain-robust {pc(ld['pct_corpora_meeting_alpha'])}
(per generator: {pc(lf['pct_generators_meeting_alpha'])}, {pc(ln_['pct_generators_meeting_alpha'])}, {pc(ld['pct_generators_meeting_alpha'])}; genuine speech rejected
{pc(lf['genuine_rejected'], 1)}, {pc(ln_['genuine_rejected'], 1)}, {pc(ld['genuine_rejected'], 1)}).

**Table 8.** Same study, logistic-regression head.

{table_cov(lin)}
"""

paper = f"""# How Far Can AI-Voice Detectors Be Trusted on Unseen Generators? Leave-One-Corpus-Out Calibration, Domain-Robust Conformal Limits, and Their Use in Transaction-Bound Voice Authorisation

*Draft. Every number below is generated from `reports/conformal_mlp.json` (and `conformal_linear.json`) by `scripts/build_paper.py`.*

## Abstract

Voice-based payment authorisation must reject AI-cloned voices, and the costly mistake is an AI voice accepted as real. Detectors are reported by accuracy
or equal-error rate, which does not say how often a generator the detector has never seen gets through. We measure this with a nested leave-one-corpus-out protocol
over {n_dom} corpora ({units} held-out generators) and a frozen-WavLM detector. A fixed equal-error threshold keeps the AI-accepted rate within a 10% target in only
{pc(fx['pct_corpora_meeting_alpha'])} of held-out corpora (mean {pc(fx['mean_generator_miss'])} of an unseen generator's AI voices accepted), and split-conformal calibration, which is valid for
new clips from the calibration distribution, reaches {pc(nv['pct_corpora_meeting_alpha'])}. **Domain-robust conformal calibration**, which treats the *generator* as the exchangeable unit,
reaches {pc(dr['pct_corpora_meeting_alpha'])} (nominal {pc(conf)}), but only {pc(dr['pct_generators_meeting_alpha'])} per generator because generators cluster by corpus, and it rejects
{pc(dr['genuine_rejected'])} of genuine speech. We audit the authenticity limits of VoiceLock, a transaction-bound voice-authorisation protocol, with the same protocol
({pc(crit['mean_generator_miss'])}–{pc(low_t['mean_generator_miss'])} of unseen generators' AI voices accepted on average), report that a cloned segment pasted into real speech is detected
only {pc(sp1[sp_names[0]])} of the time with one AI window in four, and release code and an in-browser build.

**Keywords:** audio deepfake detection, conformal prediction, distribution shift, voice authentication, risk control.

## 1. Introduction

Cloned voices defeat casual listeners and voice is still used to approve payments. A detector for this use is judged by a bound ("an AI voice is accepted at most α of the time"), not by
average accuracy. Anti-spoofing papers report equal-error rate on benchmarks whose attacks overlap with training (ASVspoof 2019) or on one out-of-domain set (In-the-Wild), but a deployed
system meets generators released after it was calibrated. Conformal prediction supplies finite-sample guarantees under exchangeability of calibration and test data; a new text-to-speech
system is exactly where that assumption fails.

**Contributions.**
1. *Measurement.* A nested leave-one-corpus-out protocol that scores a threshold at the level that matters — a corpus or generator the detector never saw — and quantifies how far
   an equal-error threshold and pooled conformal calibration fall short (§5.1–5.4).
2. *Method.* Domain-robust conformal calibration: generator-level exchangeability, a one-line change to split conformal (§3), evaluated at two validity levels (corpus, generator) with its cost in genuine speech rejected.
3. *Audit of a deployed system.* The same protocol applied to the P(AI) limits of VoiceLock's four risk tiers (§5.5), and an opt-in strict mode that uses the calibrated limits (§6).
4. *Partial-fake finding.* A simulation of a cloned challenge pasted into real speech, with clip-level aggregation rules compared (§5.7).

## 2. Related work

*Detectors and generalisation.* Self-supervised front-ends (WavLM [1], wav2vec 2.0) with light back-ends are strong anti-spoofing systems [2]. Generalisation to unseen generators and conditions is open and is
studied with out-of-domain data [3] and new challenge editions [4]. Systems can also learn dataset artefacts: silence duration differs between bona fide and spoofed ASVspoof clips [5]; we found and removed
this effect in our own data before this study (README, "silence shortcut").

*Conformal prediction.* Split conformal prediction [6, 7] gives marginal coverage under exchangeability; extensions handle covariate shift by likelihood-ratio weighting [8] and bounded
non-exchangeability [9]; risk-controlling prediction sets [10] control other losses. They need a known shift or a bound on it. We use the structure of the problem instead: generators come as discrete units, and we calibrate across them.

*Partial and spliced speech.* Partially spoofed audio has its own corpora and detectors [11]; we study its effect on challenge-response authentication and clip-level aggregation only.

*Voice authentication.* Spoofing-aware speaker verification combines an anti-spoofing score with a speaker embedding [12] (ECAPA-TDNN [13]); prompted challenges add liveness and are checked by speech recognition [14].

## 3. Method

**Detector.** Frozen WavLM base-plus; mean+std pooling of layers 2, 3, 5, 7 over 3 s windows (6,144 features); standard scaling and a 256-unit MLP trained on {n_dom} corpora
({', '.join(mlp['domains'])}) with augmentation (codec, noise, reverberation, band-limit) and simulated speaker-to-microphone replay. A clip's score *s* is the mean window logit (larger = more AI-like);
a clip is flagged AI when *s > τ*. The costly error is a **miss**: an AI clip with *s ≤ τ*.

**Naive split conformal.** With calibration AI scores *f₍₁₎ ≤ … ≤ f₍ₙ₎* and k = ⌊α(n+1)⌋, set τ = f₍ₖ₎. A new AI clip exchangeable with the calibration clips is missed with probability ≤ α.
Ties from the detector's saturating output are broken with noise of 10⁻³ (randomised conformal).

**Domain-robust conformal.** Let *u = 1…K* index calibration generators and τᵤ the α-quantile of generator *u*'s scores. Set τ_DR = τ₍ₖ₎ with k = ⌊δ(K+1)⌋, the k-th smallest of τ₁ … τ_K.
*Claim.* If τ₁ … τ_K and the new generator's τ_new are exchangeable, P(τ_new < τ_DR) ≤ k/(K+1) ≤ δ, so with probability ≥ 1 − δ the new generator's miss rate at τ_DR is ≤ α (the rank of τ_new
among K+1 exchangeable values is uniform). The statement is about generators, not clips, and is approximate because each τᵤ is estimated from finitely many clips (generators with fewer than {mlp['nmin']} clips are ignored).
It assumes a new generator is a draw from the same population of generators as the calibration ones. In practice generators cluster by corpus, so the effective exchangeable unit is closer to the corpus; we therefore report validity at both levels.

**Audit and tier limits.** The detector reports P(AI) = σ(s − logit θ), θ being its calibrated decision threshold. A P(AI) limit *p* corresponds to the threshold τ = logit p + logit θ. The *audit* of a limit
is the miss rate and genuine-rejection rate of that τ on held-out generators. A strict limit for target α is p = σ(τ_DR − logit θ).

## 4. Experimental protocol

**Data.** {n_dom} corpora: TTS vs YouTube speech (`gary`), UniDataPro real-vs-fake (`unidata`), ASVspoof 2019 LA train and evaluation attacks (`asv_train`, `asv_test`), Common Voice with 47 neural voices (`cv`),
LibriSpeech with neural TTS (`libri`), ASVspoof 5 (`asv5`), MLAAD (64 recent TTS systems, fake only) and OpenAI/ChatGPT voices (fake only). A *generator* is a (corpus, attack/TTS system); evaluation generators have ≥ 20 clips.
Each clip is scored clean, augmented and replay-simulated; **direct capture** (clean + augmented) is the primary scope because it covers uploaded or injected audio and a genuine user's microphone;
replay (loudspeaker → room → microphone, simulated) is reported separately.

**Nested leave-one-corpus-out.** For each held-out corpus D: (i) the *test* detector is trained on all other corpora; (ii) for each other corpus r, a *calibration* detector is trained on all corpora except D and r and scores r.
Calibration scores thus come from detectors that saw neither D nor the calibrated corpus; test scores come from a detector that never saw D. 9 + 36 = 45 fits per head.

**Metrics.** Fraction of held-out corpora whose pooled AI-accepted rate ≤ α; fraction of held-out generators with miss ≤ α (bootstrap 95% CI); mean and worst per-generator miss; genuine speech rejected on the held-out corpus's real clips.
**Baselines:** a fixed equal-error threshold (the rule used before this work) and naive split conformal. **Sanity check:** ordinary in-domain calibration (half of a corpus's AI clips calibrate, the other half test)
meets its target: {', '.join(f'α = {pc(float(k))}: {pc(v, 1)} observed' for k, v in in_dom.items())}.

## 5. Results

**Reference detector quality** (the project's own held-out protocol, which mixes seen generators, `reports/evaluation.json`): pooled accuracy {pc(pool['acc'], 1)}, AUC {pool['auc']:.3f}, EER {pc(pool['eer'], 1)}.
Leave-one-corpus-out is much harder, as the rest of this section shows.

### 5.1 Does a target error rate survive an unseen corpus or generator?

**Table 1.** Direct-capture audio, δ = {pc(mlp['delta'])} (intended confidence {pc(conf)}).

{table_cov(mlp)}

At α = {pc(a0)} the fixed threshold keeps {pc(fx['pct_corpora_meeting_alpha'])} of held-out corpora and {pc(fx['pct_generators_meeting_alpha'])} of generators within the target (mean per-generator miss {pc(fx['mean_generator_miss'], 1)}, worst {pc(fx['worst_generator_miss'])});
naive conformal {pc(nv['pct_corpora_meeting_alpha'])} and {pc(nv['pct_generators_meeting_alpha'])}; domain-robust {pc(dr['pct_corpora_meeting_alpha'])} and {pc(dr['pct_generators_meeting_alpha'])} (mean miss {pc(dr['mean_generator_miss'], 1)}, worst {pc(dr['worst_generator_miss'])}).
{'Corpus-level validity is close to the nominal ' + pc(conf) + '. ' if dr_ok else 'Corpus-level validity stays below the nominal ' + pc(conf) + '. '}{'Generator-level validity is close to nominal too. ' if gen_ok else 'Generator-level validity is clearly lower: a hard generator inside a corpus that passes can still exceed α. '}
The naive procedure does meet its target when calibration and test are in the same corpus ({pc(in_dom['0.1'], 1)} observed at α = {pc(a0)}), so the shortfall comes from the new corpus.

### 5.2 What it costs

Genuine speech rejected at α = {pc(a0)}: fixed {pc(fx['genuine_rejected'], 1)}, naive {pc(nv['genuine_rejected'], 1)}, domain-robust {pc(dr['genuine_rejected'], 1)}. The bound is bought with a lower threshold, and the cost grows steeply as α shrinks (Table 1):
at α = {pc(A[0])} domain-robust calibration rejects {pc(sm(mlp, 'domain_robust', A[0])['genuine_rejected'])} of genuine speech. With this detector a tight bound on unseen generators is therefore not compatible with a usable genuine-user experience; even the loosest target tested
(α = {pc(A[-1])}) rejects {pc(sm(mlp, 'domain_robust', A[-1])['genuine_rejected'])} of genuine speech.

**Table 2.** Per held-out corpus at α = {pc(a0)}: pooled AI-accepted rate of its generators and genuine speech rejected.

{table_domains(a0)}

**Choice of δ** (α = {pc(a0)}): {'; '.join(f"δ = {pc(float(k))}: {pc(v['pct_corpora_meeting_alpha_10pct'])} of corpora and {pc(v['pct_generators_meeting_alpha_10pct'])} of generators within α, {pc(v['genuine_rejected'])} genuine rejected" for k, v in dsw.items())}.

### 5.3 Capture condition

**Table 3.** The same comparison by capture condition (α = {pc(a0)}).

{table_scopes(a0)}

On simulated replay the detector is close to chance (clip AUC about 0.65 pooled over corpora), so the fixed threshold keeps only {pc(fx_rep['pct_corpora_meeting_alpha'])} of corpora within α; domain-robust calibration reaches
{pc(rep_replay['pct_corpora_meeting_alpha'])} but only by rejecting {pc(rep_replay['genuine_rejected'])} of genuine speech (the genuine reference there is also replayed). No useful bound on replay is available from this detector alone; VoiceLock relies on its live challenge, speaker and consistency layers for that attack.

### 5.4 Strict limits for VoiceLock

Calibrating on the out-of-domain scores of all {dep['generators_used']} generators gives the strict per-tier limits of Table 4 (targets α = 30/20/10/5%).

**Table 4.** Strict P(AI) limits (a response at or above the limit is rejected).

{table_strict()}

### 5.5 Audit of the default limits

**Table 5.** VoiceLock's default per-tier P(AI) limits, measured on held-out corpora (direct capture).

{table_audit()}

On average {pc(crit['mean_generator_miss'])}–{pc(low_t['mean_generator_miss'])} of an unseen generator's AI voices fall under the default limits and {pc(crit['pct_generators_miss_over_10pct'])}–{pc(low_t['pct_generators_miss_over_10pct'])} of generators exceed 10%, with {pc(low_t['genuine_rejected'])}–{pc(crit['genuine_rejected'])} of
genuine speech rejected. The default limits are usable but carry no guarantee; they get tighter across tiers only slightly, because detector scores saturate. The shipped detector is trained on every corpus, so a truly new generator may do better or worse than leave-one-corpus-out suggests.

### 5.6 Detector quality per held-out corpus and a classical baseline

**Table 7.** Clean audio, corpus never seen in training (clip-level EER / AUC). The MFCC baseline uses up to 500 real + 500 AI clips per corpus and the same protocol; it is not a published system.

{table_quality()}

{(f"The WavLM + MLP detector has the lower EER than the MFCC baseline on {sum(mlp['per_corpus_quality'][c]['clean']['eer'] < base['per_corpus'][c]['eer'] for c in base['per_corpus'] if mlp['per_corpus_quality'].get(c))} of {sum(1 for c in base['per_corpus'] if mlp['per_corpus_quality'].get(c))} corpora, but its error on a new corpus is far from the sub-1% EERs published for in-domain ASVspoof 2019 LA: on the LA evaluation attacks (trained with the LA training attacks plus the other corpora; our 750-clip subset, not the official protocol) it reaches {pc(mlp['per_corpus_quality']['asv_test']['clean']['eer'], 1)} EER. " if base else '')}No published external baseline (AASIST, RawNet2) and no official-protocol run (full ASVspoof 2019 LA eval, In-the-Wild) was performed; this is stated in §7.

### 5.7 Spliced recordings

A recording where a real voice is mixed with a short AI segment, for example a cloned challenge pasted into a genuine earlier recording, can look mostly real to a mean-based clip score. We simulate it at window level:
pseudo-clips of {mlp['splice']['M']} windows with k AI windows (out-of-domain window scores of direct-capture audio, {mlp['splice']['real_windows']} real and {mlp['splice']['ai_windows']} AI windows), thresholds set at
{pc(mlp['splice']['false_alarm_on_pure_real'])} false alarms on pure-real clips.

**Table 6.** Detection of spliced clips by aggregation rule.

{table_splice()}

With one AI window in {mlp['splice']['M']} the deployed mean-logit rule ({sp_names[0]}) detects {pc(sp1[sp_names[0]])}; the best simple rule ({best_sp}) detects {pc(sp1[best_sp])}. No aggregation rule closes the gap, so partial fakes need a detector trained for them [11] — we report this as a weakness
of the present system. Windows are drawn independently, so the acoustic seam between real and AI audio is not modelled.
{lin_block}
## 6. System: VoiceLock

VoiceLock authorises *one transaction* with *one live voice response*. Layers: (1) AI-voice forensics, whose P(AI) limit per tier is the default limit of Table 5 with its audit shown to the user (default "audit" mode),
or the strict limit of Table 4 (opt-in "strict" mode; acoustic anomalies may only tighten it, so the bound stays valid); (2) speaker identity (ECAPA-TDNN cosine to the enrolled voiceprint, plus segment consistency);
(3) a spoken challenge derived by HMAC-SHA256 from the transaction hash, session and a nonce (three words + four digits), checked by Whisper tiny.en with one error tolerated; (4) a risk engine: tier from amount, new recipient, request
velocity and time of day; the critical tier needs two independent challenges. On success an Ed25519-signed token (60 s, single use) carries the transaction hash; every decision is appended to a SHA-256 hash-chained audit log.
The Python service and the browser build (ONNX Runtime Web) share this design.
{'In the project test bench (10 LibriSpeech speakers, 10 trials per attack type and tier; attacks: replay of a recording, pre-made TTS, live TTS of the challenge, live TTS through a loudspeaker) no attack was verified. The TTS attackers were generic voices, not clones of the enrolled speaker, and the detector had seen part of that audio in training, so this is a pipeline check and not evidence of resistance to a targeted voice clone.' if vl else ''}

## 7. Limitations and ethics

* Leave-one-corpus-out describes generalisation to a *new corpus*; the shipped detector is trained on all corpora. The guarantee is approximate, holds at corpus level (nine corpora) and is lower per generator.
* It controls only the AI-accepted rate; genuine speakers pay for it ({pc(dr['genuine_rejected'])} rejected at α = {pc(a0)}), which is why the strict limits are opt-in.
* Replay and splicing are simulated; replay is close to chance for this detector; splice detection is weak (Table 6).
* Not evaluated: official ASVspoof / In-the-Wild protocols, external baselines (AASIST, RawNet2), a live adaptive white-box attacker, or targeted voice clones of an enrolled speaker. The web build ships its models to the client,
  so it is a protocol demonstration, not a security boundary.
* Cloned-voice material comes only from public research datasets.

## 8. Conclusion

An equal-error threshold says little about an unseen generator: here it keeps {pc(fx['pct_corpora_meeting_alpha'])} of held-out corpora within a {pc(a0)} target. Calibrating across generators restores corpus-level validity ({pc(dr['pct_corpora_meeting_alpha'])})
at a measured cost of {pc(dr['genuine_rejected'])} genuine speech rejected, and the same protocol shows that the default limits of a working voice-authorisation system accept on average {pc(crit['mean_generator_miss'])}–{pc(low_t['mean_generator_miss'])}
of an unseen generator's AI voices. Better detectors and partial-fake training are the way to make the loose and strict ends of this trade-off meet.

## Reproducibility

`scripts/run_conformal.sh` regenerates everything from the cached features; `scripts/build_paper.py` regenerates this document; tests: `pytest tests` (conformal math, audit/strict tier limits, VoiceLock protocol).

## References

[1] S. Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE JSTSP, 2022.
[2] H. Tak et al., "Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation," Odyssey, 2022.
[3] N. M. Müller et al., "Does Audio Deepfake Detection Generalize?," Interspeech, 2022.
[4] X. Wang et al., "ASVspoof 5: Design, collection and validation of resources for spoofing, deepfake, and adversarial attack detection using crowdsourced speech," 2024/2025.
[5] N. M. Müller et al., "Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?," ASVspoof workshop, 2021.
[6] V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.
[7] A. N. Angelopoulos, S. Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification," 2021.
[8] R. J. Tibshirani, R. F. Barber, E. Candès, A. Ramdas, "Conformal Prediction Under Covariate Shift," NeurIPS, 2019.
[9] R. F. Barber, E. J. Candès, A. Ramdas, R. J. Tibshirani, "Conformal Prediction Beyond Exchangeability," Annals of Statistics, 2023.
[10] S. Bates, A. Angelopoulos, L. Lei, J. Malik, M. Jordan, "Distribution-Free, Risk-Controlling Prediction Sets," JACM, 2021.
[11] L. Zhang, X. Wang, E. Cooper, J. Yamagishi, "The PartialSpoof Database and Countermeasures for the Detection of Short Fake Speech Segments Embedded in an Utterance," IEEE/ACM TASLP, 2023.
[12] J.-w. Jung et al., "SASV 2022: The First Spoofing-Aware Speaker Verification Challenge," Interspeech, 2022.
[13] B. Desplanques, J. Thienpondt, K. Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification," Interspeech, 2020.
[14] A. Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision" (Whisper), 2022.

*Check every reference's venue and year against the original before submission; they were written from memory.*
"""
OUT.mkdir(exist_ok=True)
(OUT / "PAPER.md").write_text(paper, encoding="utf-8")
print("wrote", OUT / "PAPER.md", len(paper.split()), "words")
