# How Far Can AI-Voice Detectors Be Trusted on Unseen Generators? Leave-One-Corpus-Out Calibration, Domain-Robust Conformal Limits, and Their Use in Transaction-Bound Voice Authorisation

*Draft. Every number below is generated from `reports/conformal_mlp.json` (and `conformal_linear.json`) by `scripts/build_paper.py`.*

## Abstract

Voice-based payment authorisation must reject AI-cloned voices, and the costly mistake is an AI voice accepted as real. Detectors are reported by accuracy
or equal-error rate, which does not say how often a generator the detector has never seen gets through. We measure this with a nested leave-one-corpus-out protocol
over 9 corpora (131 held-out generators) and a frozen-WavLM detector. A fixed equal-error threshold keeps the AI-accepted rate within a 10% target in only
44% of held-out corpora (mean 26% of an unseen generator's AI voices accepted), and split-conformal calibration, which is valid for
new clips from the calibration distribution, reaches 67%. **Domain-robust conformal calibration**, which treats the *generator* as the exchangeable unit,
reaches 89% (nominal 90%), but only 75% per generator because generators cluster by corpus, and it rejects
70% of genuine speech. We audit the authenticity limits of VoiceLock, a transaction-bound voice-authorisation protocol, with the same protocol
(23%–27% of unseen generators' AI voices accepted on average), report that a cloned segment pasted into real speech is detected
only 14% of the time with one AI window in four, and release code and an in-browser build.

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

**Detector.** Frozen WavLM base-plus; mean+std pooling of layers 2, 3, 5, 7 over 3 s windows (6,144 features); standard scaling and a 256-unit MLP trained on 9 corpora
(gary, unidata, asv_train, cv, libri, asv5, mlaad, chatgpt, asv_test) with augmentation (codec, noise, reverberation, band-limit) and simulated speaker-to-microphone replay. A clip's score *s* is the mean window logit (larger = more AI-like);
a clip is flagged AI when *s > τ*. The costly error is a **miss**: an AI clip with *s ≤ τ*.

**Naive split conformal.** With calibration AI scores *f₍₁₎ ≤ … ≤ f₍ₙ₎* and k = ⌊α(n+1)⌋, set τ = f₍ₖ₎. A new AI clip exchangeable with the calibration clips is missed with probability ≤ α.
Ties from the detector's saturating output are broken with noise of 10⁻³ (randomised conformal).

**Domain-robust conformal.** Let *u = 1…K* index calibration generators and τᵤ the α-quantile of generator *u*'s scores. Set τ_DR = τ₍ₖ₎ with k = ⌊δ(K+1)⌋, the k-th smallest of τ₁ … τ_K.
*Claim.* If τ₁ … τ_K and the new generator's τ_new are exchangeable, P(τ_new < τ_DR) ≤ k/(K+1) ≤ δ, so with probability ≥ 1 − δ the new generator's miss rate at τ_DR is ≤ α (the rank of τ_new
among K+1 exchangeable values is uniform). The statement is about generators, not clips, and is approximate because each τᵤ is estimated from finitely many clips (generators with fewer than 8 clips are ignored).
It assumes a new generator is a draw from the same population of generators as the calibration ones. In practice generators cluster by corpus, so the effective exchangeable unit is closer to the corpus; we therefore report validity at both levels.

**Audit and tier limits.** The detector reports P(AI) = σ(s − logit θ), θ being its calibrated decision threshold. A P(AI) limit *p* corresponds to the threshold τ = logit p + logit θ. The *audit* of a limit
is the miss rate and genuine-rejection rate of that τ on held-out generators. A strict limit for target α is p = σ(τ_DR − logit θ).

## 4. Experimental protocol

**Data.** 9 corpora: TTS vs YouTube speech (`gary`), UniDataPro real-vs-fake (`unidata`), ASVspoof 2019 LA train and evaluation attacks (`asv_train`, `asv_test`), Common Voice with 47 neural voices (`cv`),
LibriSpeech with neural TTS (`libri`), ASVspoof 5 (`asv5`), MLAAD (64 recent TTS systems, fake only) and OpenAI/ChatGPT voices (fake only). A *generator* is a (corpus, attack/TTS system); evaluation generators have ≥ 20 clips.
Each clip is scored clean, augmented and replay-simulated; **direct capture** (clean + augmented) is the primary scope because it covers uploaded or injected audio and a genuine user's microphone;
replay (loudspeaker → room → microphone, simulated) is reported separately.

**Nested leave-one-corpus-out.** For each held-out corpus D: (i) the *test* detector is trained on all other corpora; (ii) for each other corpus r, a *calibration* detector is trained on all corpora except D and r and scores r.
Calibration scores thus come from detectors that saw neither D nor the calibrated corpus; test scores come from a detector that never saw D. 9 + 36 = 45 fits per head.

**Metrics.** Fraction of held-out corpora whose pooled AI-accepted rate ≤ α; fraction of held-out generators with miss ≤ α (bootstrap 95% CI); mean and worst per-generator miss; genuine speech rejected on the held-out corpus's real clips.
**Baselines:** a fixed equal-error threshold (the rule used before this work) and naive split conformal. **Sanity check:** ordinary in-domain calibration (half of a corpus's AI clips calibrate, the other half test)
meets its target: α = 5%: 4.2% observed, α = 10%: 9.3% observed, α = 20%: 18.4% observed, α = 30%: 28.0% observed, α = 40%: 38.6% observed.

## 5. Results

**Reference detector quality** (the project's own held-out protocol, which mixes seen generators, `reports/evaluation.json`): pooled accuracy 87.8%, AUC 0.951, EER 12.2%.
Leave-one-corpus-out is much harder, as the rest of this section shows.

### 5.1 Does a target error rate survive an unseen corpus or generator?

**Table 1.** Direct-capture audio, δ = 10% (intended confidence 90%).

| target α | method | corpora within α | generators within α (95% CI) | mean miss / generator | worst generator | genuine speech rejected |
|---|---|---|---|---|---|---|
| 5% | fixed equal-error threshold | 22% of 9 | 22% [15%, 30%] | 26.0% | 89% | 16.1% |
| 5% | naive split conformal | 67% of 9 | 56% [48%, 65%] | 9.5% | 65% | 50.2% |
| 5% | **domain-robust (ours)** | 89% of 9 | 71% [63%, 79%] | 4.4% | 46% | 81.5% |
| 10% | fixed equal-error threshold | 44% of 9 | 39% [31%, 47%] | 26.0% | 89% | 16.1% |
| 10% | naive split conformal | 67% of 9 | 53% [45%, 62%] | 14.9% | 81% | 35.0% |
| 10% | **domain-robust (ours)** | 89% of 9 | 75% [67%, 82%] | 6.5% | 54% | 70.0% |
| 20% | fixed equal-error threshold | 56% of 9 | 49% [40%, 57%] | 26.0% | 89% | 16.1% |
| 20% | naive split conformal | 78% of 9 | 52% [44%, 60%] | 26.1% | 97% | 20.4% |
| 20% | **domain-robust (ours)** | 100% of 9 | 78% [71%, 85%] | 10.4% | 68% | 55.0% |
| 30% | fixed equal-error threshold | 78% of 9 | 62% [53%, 70%] | 26.0% | 89% | 16.1% |
| 30% | naive split conformal | 67% of 9 | 50% [41%, 59%] | 36.8% | 97% | 11.9% |
| 30% | **domain-robust (ours)** | 100% of 9 | 81% [74%, 88%] | 13.0% | 76% | 45.9% |
| 40% | fixed equal-error threshold | 89% of 9 | 71% [63%, 79%] | 26.0% | 89% | 16.1% |
| 40% | naive split conformal | 67% of 9 | 49% [40%, 58%] | 47.5% | 100% | 6.9% |
| 40% | **domain-robust (ours)** | 100% of 9 | 77% [69%, 84%] | 19.7% | 86% | 38.0% |

At α = 10% the fixed threshold keeps 44% of held-out corpora and 39% of generators within the target (mean per-generator miss 26.0%, worst 89%);
naive conformal 67% and 53%; domain-robust 89% and 75% (mean miss 6.5%, worst 54%).
Corpus-level validity is close to the nominal 90%. Generator-level validity is clearly lower: a hard generator inside a corpus that passes can still exceed α. 
The naive procedure does meet its target when calibration and test are in the same corpus (9.3% observed at α = 10%), so the shortfall comes from the new corpus.

### 5.2 What it costs

Genuine speech rejected at α = 10%: fixed 16.1%, naive 35.0%, domain-robust 70.0%. The bound is bought with a lower threshold, and the cost grows steeply as α shrinks (Table 1):
at α = 5% domain-robust calibration rejects 82% of genuine speech. With this detector a tight bound on unseen generators is therefore not compatible with a usable genuine-user experience; even the loosest target tested
(α = 40%) rejects 38% of genuine speech.

**Table 2.** Per held-out corpus at α = 10%: pooled AI-accepted rate of its generators and genuine speech rejected.

| held-out corpus | domain-robust τ (mean logit) | AI accepted, domain-robust (naive; fixed) | genuine rejected, domain-robust |
|---|---|---|---|
| gary | -8.34 | 1.7% (3.0%; 7.7%) | 79.8% |
| unidata | -7.50 | 1.0% (27.5%; 68.6%) | 69.4% |
| asv_train | -6.95 | 1.0% (5.0%; 15.4%) | 49.6% |
| cv | -7.62 | 0.2% (0.8%; 3.9%) | 83.5% |
| libri | -5.98 | 0.0% (0.0%; 0.7%) | 79.7% |
| asv5 | -7.74 | 0.7% (7.7%; 28.6%) | 82.6% |
| mlaad | -5.43 | 12.1% (24.4%; 35.2%) | n/a (no real speech) |
| chatgpt | -8.29 | 0.1% (2.2%; 9.5%) | n/a (no real speech) |
| asv_test | -7.93 | 2.9% (11.7%; 25.5%) | 45.1% |

**Choice of δ** (α = 10%): δ = 5%: 100% of corpora and 79% of generators within α, 79% genuine rejected; δ = 10%: 89% of corpora and 75% of generators within α, 70% genuine rejected; δ = 20%: 89% of corpora and 66% of generators within α, 58% genuine rejected.

### 5.3 Capture condition

**Table 3.** The same comparison by capture condition (α = 10%).

| scope (α = 10%) | method | corpora within α | generators within α | genuine speech rejected |
|---|---|---|---|---|
| direct capture (clean + codec/noise/reverb/band-limit) | fixed equal-error threshold | 44% | 39% | 16.1% |
| direct capture (clean + codec/noise/reverb/band-limit) | naive split conformal | 67% | 53% | 35.0% |
| direct capture (clean + codec/noise/reverb/band-limit) | **domain-robust (ours)** | 89% | 75% | 70.0% |
| all conditions incl. simulated replay | fixed equal-error threshold | 33% | 27% | 20.1% |
| all conditions incl. simulated replay | naive split conformal | 56% | 55% | 47.2% |
| all conditions incl. simulated replay | **domain-robust (ours)** | 89% | 79% | 72.9% |
| simulated loudspeaker → microphone replay only | fixed equal-error threshold | 11% | 15% | 31.3% |
| simulated loudspeaker → microphone replay only | naive split conformal | 56% | 60% | 73.8% |
| simulated loudspeaker → microphone replay only | **domain-robust (ours)** | 100% | 83% | 91.5% |

On simulated replay the detector is close to chance (clip AUC about 0.65 pooled over corpora), so the fixed threshold keeps only 11% of corpora within α; domain-robust calibration reaches
100% but only by rejecting 92% of genuine speech (the genuine reference there is also replayed). No useful bound on replay is available from this detector alone; VoiceLock relies on its live challenge, speaker and consistency layers for that attack.

### 5.4 Strict limits for VoiceLock

Calibrating on the out-of-domain scores of all 144 generators gives the strict per-tier limits of Table 4 (targets α = 30/20/10/5%).

**Table 4.** Strict P(AI) limits (a response at or above the limit is rejected).

| tier | target α | strict P(AI) limit | genuine speech rejected (pooled) |
|---|---|---|---|
| low | 30% | 0.346% | 53.1% |
| medium | 20% | 0.081% | 60.6% |
| high | 10% | 0.007% | 73.8% |
| critical | 5% | 0.002% | 87.4% |

### 5.5 Audit of the default limits

**Table 5.** VoiceLock's default per-tier P(AI) limits, measured on held-out corpora (direct capture).

| tier | default P(AI) limit | AI accepted, mean over unseen generators | generators with > 10% accepted | genuine speech rejected |
|---|---|---|---|---|
| low | 50% | 26.9% | 61% | 21.7% |
| medium | 45% | 26.1% | 60% | 22.6% |
| high | 35% | 24.7% | 59% | 24.1% |
| critical | 25% | 22.9% | 56% | 26.2% |

On average 23%–27% of an unseen generator's AI voices fall under the default limits and 56%–61% of generators exceed 10%, with 22%–26% of
genuine speech rejected. The default limits are usable but carry no guarantee; they get tighter across tiers only slightly, because detector scores saturate. The shipped detector is trained on every corpus, so a truly new generator may do better or worse than leave-one-corpus-out suggests.

### 5.6 Reference baselines

No external baselines (AASIST, RawNet2) or official-protocol numbers (ASVspoof 2019 LA eval, In-the-Wild) were run; this is stated in §7.

### 5.7 Spliced recordings

A recording where a real voice is mixed with a short AI segment, for example a cloned challenge pasted into a genuine earlier recording, can look mostly real to a mean-based clip score. We simulate it at window level:
pseudo-clips of 4 windows with k AI windows (out-of-domain window scores of direct-capture audio, 5250 real and 12397 AI windows), thresholds set at
5% false alarms on pure-real clips.

**Table 6.** Detection of spliced clips by aggregation rule.

| aggregation rule | 1/4 windows AI | 2/4 windows AI | 3/4 windows AI | 4/4 windows AI |
|---|---|---|---|---|
| mean (deployed) | 14% | 35% | 64% | 86% |
| median | 15% | 36% | 65% | 84% |
| max | 14% | 22% | 30% | 37% |
| mean of top-2 | 17% | 45% | 67% | 82% |

With one AI window in 4 the deployed mean-logit rule (mean (deployed)) detects 14%; the best simple rule (mean of top-2) detects 17%. No aggregation rule closes the gap, so partial fakes need a detector trained for them [11] — we report this as a weakness
of the present system. Windows are drawn independently, so the acoustic seam between real and AI audio is not modelled.

### 5.8 Is it specific to the MLP head?

Repeating the whole nested protocol with a logistic-regression head on the same features gives the same ordering (Table 8): at α = 10% the fixed threshold keeps
22% of held-out corpora within the target, naive conformal 78%, domain-robust 100%
(per generator: 30%, 55%, 86%; genuine speech rejected
17.5%, 43.6%, 81.4%).

**Table 8.** Same study, logistic-regression head.

| target α | method | corpora within α | generators within α (95% CI) | mean miss / generator | worst generator | genuine speech rejected |
|---|---|---|---|---|---|---|
| 5% | fixed equal-error threshold | 11% of 9 | 15% [9%, 21%] | 30.1% | 86% | 17.5% |
| 5% | naive split conformal | 67% of 9 | 56% [47%, 64%] | 8.2% | 61% | 63.3% |
| 5% | **domain-robust (ours)** | 89% of 9 | 83% [76%, 89%] | 2.6% | 32% | 89.2% |
| 10% | fixed equal-error threshold | 22% of 9 | 30% [22%, 37%] | 30.1% | 86% | 17.5% |
| 10% | naive split conformal | 78% of 9 | 55% [46%, 63%] | 14.9% | 76% | 43.6% |
| 10% | **domain-robust (ours)** | 100% of 9 | 86% [80%, 92%] | 5.0% | 42% | 81.4% |
| 20% | fixed equal-error threshold | 56% of 9 | 50% [42%, 59%] | 30.1% | 86% | 17.5% |
| 20% | naive split conformal | 67% of 9 | 50% [42%, 59%] | 26.6% | 86% | 26.5% |
| 20% | **domain-robust (ours)** | 100% of 9 | 81% [74%, 88%] | 9.7% | 66% | 66.8% |
| 30% | fixed equal-error threshold | 67% of 9 | 55% [47%, 63%] | 30.1% | 86% | 17.5% |
| 30% | naive split conformal | 67% of 9 | 50% [41%, 58%] | 36.9% | 97% | 15.6% |
| 30% | **domain-robust (ours)** | 100% of 9 | 84% [77%, 90%] | 13.8% | 76% | 57.5% |
| 40% | fixed equal-error threshold | 78% of 9 | 61% [53%, 69%] | 30.1% | 86% | 17.5% |
| 40% | naive split conformal | 67% of 9 | 46% [37%, 54%] | 47.3% | 97% | 9.2% |
| 40% | **domain-robust (ours)** | 100% of 9 | 84% [77%, 90%] | 18.2% | 81% | 45.9% |

## 6. System: VoiceLock

VoiceLock authorises *one transaction* with *one live voice response*. Layers: (1) AI-voice forensics, whose P(AI) limit per tier is the default limit of Table 5 with its audit shown to the user (default "audit" mode),
or the strict limit of Table 4 (opt-in "strict" mode; acoustic anomalies may only tighten it, so the bound stays valid); (2) speaker identity (ECAPA-TDNN cosine to the enrolled voiceprint, plus segment consistency);
(3) a spoken challenge derived by HMAC-SHA256 from the transaction hash, session and a nonce (three words + four digits), checked by Whisper tiny.en with one error tolerated; (4) a risk engine: tier from amount, new recipient, request
velocity and time of day; the critical tier needs two independent challenges. On success an Ed25519-signed token (60 s, single use) carries the transaction hash; every decision is appended to a SHA-256 hash-chained audit log.
The Python service and the browser build (ONNX Runtime Web) share this design.
In the project test bench (10 LibriSpeech speakers, 10 trials per attack type and tier; attacks: replay of a recording, pre-made TTS, live TTS of the challenge, live TTS through a loudspeaker) no attack was verified. The TTS attackers were generic voices, not clones of the enrolled speaker, and the detector had seen part of that audio in training, so this is a pipeline check and not evidence of resistance to a targeted voice clone.

## 7. Limitations and ethics

* Leave-one-corpus-out describes generalisation to a *new corpus*; the shipped detector is trained on all corpora. The guarantee is approximate, holds at corpus level (nine corpora) and is lower per generator.
* It controls only the AI-accepted rate; genuine speakers pay for it (70% rejected at α = 10%), which is why the strict limits are opt-in.
* Replay and splicing are simulated; replay is close to chance for this detector; splice detection is weak (Table 6).
* Not evaluated: official ASVspoof / In-the-Wild protocols, external baselines (AASIST, RawNet2), a live adaptive white-box attacker, or targeted voice clones of an enrolled speaker. The web build ships its models to the client,
  so it is a protocol demonstration, not a security boundary.
* Cloned-voice material comes only from public research datasets.

## 8. Conclusion

An equal-error threshold says little about an unseen generator: here it keeps 44% of held-out corpora within a 10% target. Calibrating across generators restores corpus-level validity (89%)
at a measured cost of 70% genuine speech rejected, and the same protocol shows that the default limits of a working voice-authorisation system accept on average 23%–27%
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
