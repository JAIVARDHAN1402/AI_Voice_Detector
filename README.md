# AI Voice Detector

Tells you whether a voice is a **real human** or **AI-generated** — from an uploaded audio file or live from your microphone.

```
audio (wav/mp3/m4a/flac/ogg/webm, or mic)
  -> 16 kHz mono, trim silence, 3 s windows
  -> WavLM (frozen, lower 8 layers): mean+std over time of layers 2,3,5,7
  -> standardise + small neural network (one 256-unit layer)  -> P(AI) per window
  -> average in logit space, calibrated threshold, data-driven verdict band -> Real / AI-generated / Uncertain
```

## Run it

Requires Python 3.11 and `ffmpeg` on PATH (used to decode m4a/webm/mp3 and for mic recordings).

```bash
.venv\Scripts\python.exe app\server.py          # then open http://127.0.0.1:8008
.venv\Scripts\python.exe predict.py clip.wav    # command line
```

Set `PORT=xxxx` to use another port. The trained model is `models/detector.joblib`; the WavLM backbone
is in `models/wavlm-base-plus/` and is **not** in git (378 MB) - after cloning, download it once:

```bash
bash scripts/fetch.sh https://huggingface.co/microsoft/wavlm-base-plus/resolve/main/pytorch_model.bin models/wavlm-base-plus/pytorch_model.bin
```

To set up from scratch: `pip install -r requirements.txt`, then run the
scripts in the order below.

## How good is it? (honest numbers)

**Latest model** (`reports/evaluation.json`; after removing the silence shortcut, tuning on the 18 user segments, ChatGPT voices):
pooled held-out accuracy **87.8 %**, AUC 0.951 across clean, augmented and speaker-to-mic audio. With the verdict band
(real <= 0.22, AI >= 0.50): AI called "real" 8 %, real called "AI" 12 %, real called real 82 %, AI called AI 88 %.
New generators never trained on: MLAAD 82.5 %, ASVspoof 5 A29-A32 89.5 %, ChatGPT voices (onyx, shimmer, app Juniper) 92 %.
Independent `ood_check.py`: LibriSpeech test-clean real 96 % correct / 4 % wrong, Windows + Google TTS 100 %.
The tables below were measured on earlier versions of the model; they show the shape of the results, not the exact current values.

### The silence shortcut (fixed)

`asv_train`, `asv_test` and `unidata` kept long leading/trailing silence, and its length differed between real and fake
(asv_test: real 1.4 s vs fake 0.6 s). The model partly learned "less silence = fake". The app trims silence, so on the very same
real ASVspoof clips the production pipeline said "AI" for 83 % of them while a training-style window said "real". These three
sets are now trimmed exactly like the app does (`scripts/extract_features.py: TRIM_SETS`), features re-extracted, model retrained.
Production-pipeline acceptance of those real clips: 17 % -> 98 %. Pooled held-out accuracy fell from 89.4 % to 87.8 % - the old
number was partly the shortcut.

Everything is evaluated on **speakers/recordings the model never saw** during that evaluation. "Augmented" = the same
test audio pushed through MP3/AAC/Opus codecs, noise, reverb or band-limiting (closer to a phone or browser mic).

| Test set | Clean acc. | Augmented acc. |
|---|---|---|
| Modern TTS vs YouTube speech (garystafford) | 95.8 % | 88.3 % |
| Neural TTS (47 voices) vs Common Voice volunteers (368 speakers) | 98.4 % | 93.1 % |
| Neural TTS vs LibriSpeech readers | 98.1 % | 94.0 % |
| ASVspoof 2019 (known attacks, new speakers) | 98.7 % | 95.7 % |
| **ASVspoof 2019 — attacks A07–A19 never trained on** | **86.0 %** (AUC 0.94) | 82.2 % |

(Numbers in this first table are from the earlier model; the sections below cover the current one: pooled held-out 88.6 % accuracy, AUC 0.955 across clean, augmented and replay audio.)

**A separate check with generators and speakers that never touched training or threshold selection**
(`scripts/ood_check.py`):

| Audio | Correct | Uncertain | Wrong |
|---|---|---|---|
| Real: LibriSpeech test-clean readers (48 clips) | 81 % | 8 % | 10 % |
| Fake: Windows built-in voices (David/Zira/Hazel) | 100 % | 0 % | 0 % |
| Fake: Google TTS (5 accents) | 100 % | 0 % | 0 % |

Re-encoded as WebM/Opus 32 kbps (what a browser microphone produces): real 79 / 8 / 12 %, both fake sets 100 %.

### Recent TTS systems (MLAAD, 64 generators, 2024-2025)

"AI voice shows 27 % and is called real" had a simple cause: the previous model caught only **60 %** of output from 64 recent
TTS / voice-cloning systems (Index-TTS-2.0 22 %, VibeVoice 31-40 %, DeepGram 33 %, Cartesia Sonic-3 45 %, ...; 24 of 64
generators were detected less than half the time). Training now includes ~1,500 clips from all 64 systems
([MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny), CC BY-NC 4.0; Cartesia, Chatterbox, Higgs-Audio, VibeVoice,
Qwen2.5-Omni, MiniMax, Sesame, F5, Fish, Kyutai, Resemble, ...), a small neural-network head instead of logistic regression
(+2 AUC points, +4 points of AI caught at 10 % false alarms when a whole dataset is held out), and a verdict band chosen from held-out
scores rather than a fixed margin.

| Held-out | Result |
|---|---|
| AI detected on generators trained on (held-out clips) | 90 % |
| **AI detected on 12 newest generators never trained on** (Cartesia Sonic-3, Higgs-Audio-V2, VibeVoice Large, Index-TTS-2.0, DeepGram, Sesame, Chatterbox, MiniMax 2.6, Qwen2.5-Omni, Maya1, Kyutai, SoulX) | **86 %** |
| ASVspoof 5 clean / never-trained systems A29-A32 | 96 % / 94 % |
| Verdict policy on held-out windows: AI called "real" / real called "AI" | 8 % / 8 % (see note below: "AI" band now starts at 0.50) |
| Real voices called real / AI called AI | 86 % / 87 % |
| Independent check: unseen LibriSpeech readers correct / wrong; Windows + Google TTS | 88 % / 10 %; 100 % |

### Verdict band (user-lowered)

After testing real ChatGPT voice audio that scored 41-53 % and stayed "uncertain", the "AI" boundary (`ai_min`) was lowered by hand
from 0.707 to **0.50** in `models/detector.joblib` / `web/model/classifier.json` (re-run `scripts/export_classifier.py` after changing it;
`train.py final` recomputes it from quantiles, so set it again after a retrain). `real_max` stays 0.281, so 0.28-0.50 is "uncertain".
Held-out cost: real windows called AI rise from ~8 % to ~10.5 %; AI windows called AI rise from 87 % to ~89.5 %.

### ChatGPT / OpenAI voices

Training also includes ~1,000 clips of OpenAI voices (`scripts/prepare_chatgpt.py`: GPT-4o TTS alloy/echo/fable/nova/onyx/shimmer,
gpt-4o-mini-tts, and clean reads of the ChatGPT app voices Juniper and Sky; the app's screen recordings are not used because they may
contain the user's own voice). The previous model already caught 98 % of these as plain files but only 76 % after a simulated
speaker-to-microphone pass. Held out: voices trained on 99 %, voices never trained on (onyx, shimmer, app Juniper) 91 % (Juniper alone 81 %).
ChatGPT *voice mode* streamed through a phone or speaker is not in the data (no public recordings), so it can still score low; use the
"Teach it" tab plus `scripts/tune_user.py` for that.

Weakest generators: MegaTTS3 33%, Llasa-8B 64%, parler_tts_mini_v0.1 64%, suno_bark-small 64%, zonosTTS-v0.1 67%, Kyutai-TTS* 70%, Kani-TTS-370M 71%, sesame_csm* 72% (* never trained on). Real voices are flagged as AI a bit more often than before (unseen LibriSpeech
4 % -> 10 %): that is the price of catching more modern AI. The browser version uses an int8-shrunk backbone, so borderline clips can
read differently from the local Python app (about 3 % of individual 3 s segments flip).

### Modern generators (ASVspoof 5, 2024)

Real-world feedback: "AI voices are read as real". Measuring it on ASVspoof 5 (16 modern TTS/voice-conversion systems,
11 codec conditions, audiobook bona fide) showed why: the earlier model caught only **66 %** of modern AI clips on clean audio,
**43 %** after speaker->mic degradation, and some systems barely at all (A30 24 %, A32 28 %).
Adding a 2,900-clip ASVspoof 5 slice (and degraded copies) to training changed that on held-out clips:

| ASVspoof 5, held-out | Before | After |
|---|---|---|
| AI detected, clean | 66 % | 93 % |
| AI detected, speaker->mic replay | 43 % | 78 % |
| AI detected on 4 systems never trained on (A29-A32) | 24-90 % (mean ~50 %) | 87-91 % |
| Real voices kept as real (clean) | 72 % | 91 % |

Independent check (never used in training): unseen LibriSpeech readers 92 % correct / 4 % wrong (was 81 % / 10 %);
Windows and Google TTS still 100 %. Caveats: ASVspoof speakers can appear in both train and test (as AI vs bona fide),
which flatters the held-out numbers somewhat, and today's commercial systems (ElevenLabs v3, ChatGPT/Gemini voices,
cloning services) are newer than 2024 research systems - expect misses there. Heavily re-recorded audio is still ~50 % for
older-style generators.

### Speaker-to-microphone ("replay") audio

A live mic hears the room, not the file. Playing an AI voice through speakers and recording it destroys many of the
artefacts the model relies on, and the first version of this model called most such audio **real** (simulated test: only
10-30 % of heavily degraded AI clips were caught, 45-80 % were called real). Training now includes simulated
speaker -> room reverb -> mic noise -> Opus/AAC copies of every window (`augment.replay`, `extract_replay.py`).

| Held-out set | Replay-condition accuracy |
|---|---|
| Common Voice vs neural TTS | 93 % |
| ASVspoof 2019 (known attacks) | 86 % |
| Modern TTS vs YouTube | 85 % |
| LibriSpeech vs neural TTS | 82 % |
| ASVspoof 2019 unseen attacks | 79 % |

On generators never used in training (heavy degradation): Windows voices 30 % -> 85 % detected, Google TTS 10 % -> 45 %,
neural voices 10 % -> 50 %; "AI called real" fell from 45-80 % to 0-35 %. The price: about 1-2 points on clean audio and
slightly more real voices flagged as AI (clean LibriSpeech false alarms 15 % -> 20 %). This is *simulated* replay; the
reliable fix is to record your own speaker/room/mic setup in the app's "Teach it" tab and run `tune_user.py`.

### Limits you should know about

- **Re-recorded AI audio is still the hardest case** (see above).
- **It can flag real voices as AI**, especially recordings unlike its training data. Before I added LibriSpeech
  speakers to training, 35 % of unseen real voices were wrongly flagged; it is now ~10 %. Your own microphone and room
  are an unknown: please try it on your voice and tell me what you see.
- **Hard fake types are missed.** On ASVspoof's voice-conversion/hard attacks (A17–A19) detection was only ~40 %.
  Voice *cloning* services newer than the training data may also slip through.
- **Cross-corpus transfer is weak.** When I held out an entire dataset and trained only on the others, AUC dropped
  (0.89–0.94 for modern-TTS corpora, near chance for the 2019 ASVspoof set). The detector learns the kinds of
  artefacts present in its training data; broader data is the main way to improve it.
- Short clips (< 5 s), heavy compression, and noisy rooms lower accuracy. The app says "Uncertain" in the grey zone
  and warns when segments disagree. It is a probabilistic aid, **not proof**.

## Training data (all public)

| Name | Real | Fake | Clips used |
|---|---|---|---|
| `gary` — [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) | 14 YouTube recordings | Polly, ElevenLabs, Kokoro, Hume, Luvvoice, Speechify | 1,300 |
| `asv_train` / `asv_test` — [ASVspoof 2019 LA](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA) | VCTK studio speech | attacks A01–A06 (train) / A07–A19 (test) | 1,400 / 750 |
| `unidata` — [UniDataPro real-vs-fake](https://huggingface.co/datasets/UniDataPro/real-vs-fake-human-voice-deepfake-audio) | phone-style recordings | synthetic clones | 80 |
| `cv` — [Common Voice 17 (en)](https://huggingface.co/datasets/fixie-ai/common_voice_17_0) | 420 volunteer clips, 368 speakers | same sentences via 47 Microsoft neural voices (`edge-tts`) | 839 |
| `asv5` — [ASVspoof 5 eval](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5) | MLS audiobook speakers | modern TTS/VC attacks A17-A32, 11 codecs | 1,460 |
| `mlaad` — [MLAAD-tiny](https://huggingface.co/datasets/mueller91/MLAAD-tiny) | (fake only; real from the sets above) | 64 recent TTS/cloning systems | 1,493 |
| `chatgpt` — [GPT-4o TTS voices](https://huggingface.co/datasets/ata990/AdvBench-Audio-GPT4o-TTS), [gpt-4o-mini-tts](https://huggingface.co/datasets/LALM-emotional-damage/gpt-4o-mini-tts), [ChatGPT app voices](https://huggingface.co/datasets/leafspark/openai-voices) | (fake only) | OpenAI / ChatGPT voices (alloy, echo, fable, nova, onyx, shimmer, Juniper, Sky) | 1,013 |
| `libri` — [LibriSpeech train-clean-100](https://huggingface.co/datasets/openslr/librispeech_asr) | 20 readers | same text via `edge-tts` | 481 |

The neural-TTS fakes (`cv`, `libri`) are generated with the `edge-tts` package, which sends the sentence text to
Microsoft's online speech service. Dataset licences apply to the data you download (UniDataPro is CC BY-NC-ND,
i.e. non-commercial).

## Pipeline scripts

```
scripts/fetch.sh URL OUT                     resumable curl download (WavLM weights, garystafford parquet files)
scripts/download_unidata.py                  UniDataPro snapshot
scripts/prepare_data.py <name>               gary, unidata, asv_train, asv_test -> 16 kHz FLAC + data/meta/*.csv
scripts/prepare_cv.py, prepare_libri.py      real clips + matching neural-TTS fakes
scripts/extract_features.py <dataset>        WavLM embeddings (+ codec/noise/reverb augmented copies) -> data/features/
scripts/train.py explore                     leave-one-dataset-out study (layer choice)
scripts/train.py final --layers 2 3 5 7      hold-out evaluation -> reports/evaluation.json, refit -> models/detector.joblib
scripts/ood_check.py                         out-of-distribution sanity check -> reports/ood_check.json
```

Design notes: speakers/recordings are held out as whole groups (never split inside a speaker); both classes get the
same codecs and silence trimming so "was compressed" or "starts with silence" can't be a shortcut; only the lower
8 WavLM layers are used because deepfake artefacts live in the lower/mid layers and it halves CPU cost.

## Web version (Vercel, runs fully in the browser)

`web/` is a static site: the same detector, but WavLM runs **in the visitor's browser** through ONNX Runtime Web (WASM),
so no server is needed and audio never leaves the device. Vercel cannot host the Python/torch backend (PyTorch alone is
530 MB; function limit 250 MB), which is why the model was converted.

```
web/index.html, app.js   UI (upload + microphone)
web/detector.js, dsp.js  JavaScript port of the pipeline (trim, windows, VAD, classifier, aggregation)
web/model/               wavlm_pooled.<hash>.onnx (69 MB, 7 transformer layers: int8 matmuls + fp16-stored convs), head.<hash>.bin (6.3 MB MLP head), classifier.json
web/ort/                 vendored onnxruntime-web 1.30.0 (MIT)
web/vercel.json          cross-origin-isolation headers (multi-threaded WASM) + caching
```

**Load time / stalled downloads.** The model is 69 MB. `web/vl/loader.js` downloads it in 4 MB pieces, 6 in parallel, aborts and retries a piece
that makes no progress for 15 s (the "stuck at 4/46 MB" case), and keeps the finished file in Cache Storage (ranged responses are not
cached by the HTTP cache), so later visits are instant. The site keeps the full 7-layer model on purpose: a 4-layer variant
(`models/detector_web.joblib`, 48 MB, `scripts/layer_study.py`: 86.9 % vs 87.8 % pooled) missed a short TTS challenge clip that the
7-layer model catches (P(AI) 0.08 vs 0.87), which matters for VoiceLock's short responses. It is not deployed.

Rebuild after retraining: `export_onnx.py` -> `quantize_onnx.py` -> `export_classifier.py` (writes the hashed model name and the folded
classifier; the MLP parity was checked separately, 80 windows: 0 flipped decisions). Test locally with `python scripts/dev_web.py`
(same headers as Vercel) and deploy by pushing to `main` (Vercel Git integration) or `npx vercel --prod`.

Checks: the ONNX export matches PyTorch to ~2e-6; int8 quantisation shifts window logits by 0.6 (std) with 0 flipped
decisions on 80 windows; the browser pipeline gave the same verdicts as the Python detector on 10/10 test files and identical
clip durations (P(AI) within a few points; ~0.4 s per 3 s segment on a laptop).
The "Teach it" tab and `tune_user.py` are part of the local Flask app only.

## VoiceLock: transaction-bound voice authorisation

A voice detector alone is not a security control: a cloned voice that passes it still sounds like the CEO on the phone.
VoiceLock (`voicelock/`, `python -m voicelock` -> http://127.0.0.1:8010) makes the detector one layer of an authorisation flow in
which a recorded or pre-generated voice cannot approve an action.

```
request (user, amount, recipient)  ->  risk tier + a challenge derived from THIS transaction
spoken response  ->  1 AI-voice forensics   (this project's detector + acoustic checks)
                 ->  2 speaker identity     (ECAPA-TDNN cosine to the enrolled voiceprint, per-segment consistency)
                 ->  3 transaction challenge (Whisper: did they say the phrase for this transaction?)
                 ->  4 risk engine           (ALL layers must pass; tier tightens thresholds; critical = 2 independent challenges)
verified -> single-use Ed25519 token bound to the transaction hash      blocked -> security incident in the audit log
```

* **Challenge** = 3 words + 4 digits from `HMAC-SHA256(server secret, txn hash, session id, nonce, pass)`: unpredictable, valid for
  90 s, answerable once, different for every transaction/session. 2.9 billion phrases.
* **Token** = JWS (EdDSA) with the SHA-256 of the canonical transaction (user, amount, currency, recipient, reference); it expires
  in 60 s and is consumed atomically on first use. `POST /v1/demo/transfer` is a mock payment API that refuses anything without a valid token.
* **Audit log** = SQLite rows hash-chained with SHA-256 (`/v1/events/verify` detects an edited, deleted or reordered event). Audio is never stored, only a hash.
* **Risk tiers** (amount, new recipient, request velocity, off-hours) set the P(AI) limit (0.50 -> 0.25), the speaker threshold (0.40 -> 0.55) and the number of challenges (1 -> 2).
* Layer failures give a fresh challenge (3 failed attempts block); an almost-certainly-synthetic response (P(AI) >= 0.9) or a voice that is plainly someone else blocks immediately.
* API: `POST /v1/users/{id}/enroll`, `POST /v1/transactions`, `POST /v1/sessions/{id}/respond`, `POST /v1/tokens/verify`, `POST /v1/demo/transfer`, `GET /v1/events`, `GET /v1/events/verify`, Swagger at `/docs`. Optional `VOICELOCK_API_KEY`.
* Models (ECAPA 85 MB, Whisper tiny.en 150 MB) load from `models/voicelock/` or are downloaded from Hugging Face on first use. Tests: `pytest tests/test_voicelock.py` (18 protocol tests with stubbed models).

### What was measured (`scripts/voicelock_eval.py`, `reports/voicelock_eval.json`)

| Experiment | Result |
|---|---|
| Speaker verification, 64 speakers, 4 s clips, 3 enrolment samples | EER **0.65 %**; at the tier thresholds 0.40/0.45/0.50/0.55: false accept 1.6 / 0.6 / 0.09 / 0.02 %, false reject 0.7 / 0.7 / 0.7 / 2.6 % |
| ...with noise added to the genuine speaker | false reject at SNR 10 dB: 3 / 10 / 32 / 52 %; at 5 dB: 20 / 40 / 58 / 80 % (low / medium / high / critical) |
| Challenge recognition (TTS reading the phrase) | accepted clean 96 %, SNR 10 dB 92 %, SNR 5 dB 83 %, loudspeaker->mic simulation **69 %**; a wrong phrase accepted 0 / 48 |
| Token tampering, 2000 trials each (payload bit flip, signature bit flip, re-signed with another key, other amount/recipient) | **0 accepted**; expired token rejected |
| End to end, 10 LibriSpeech speakers, low and high tier | genuine user verified 10/10 and 10/10; replay of someone's recording, pre-made TTS, live TTS of the challenge, and live TTS played through a loudspeaker: **0 verified** in every case (40 attack trials per tier). Median latency 3.0-4.6 s on CPU |
| Layer 1 alone, 239 AI clips (ASVspoof 2019 + 5, MLAAD, OpenAI voices, Microsoft voices) vs 138 real | AI accepted 1.3 %, real accepted 99 %, EER 0.8 % (**these clips were in the detector's training data**: a pipeline check, not a generalisation number) |

### VoiceLock on the website (https://ai-voice-detector.vercel.app/voicelock)

The Vercel site cannot run the Python backend, so `web/voicelock.html` + `web/vl/` run the **same four layers in the browser**:
the site's WavLM detector, **ECAPA-TDNN exported to ONNX** (`scripts/export_ecapa_onnx.py`; embeddings match SpeechBrain with cosine
1.00000 on 12 clips, 44 MB), **Whisper tiny.en** through transformers.js (self-hosted, 41 MB, int8), the HMAC challenge, risk engine,
signed single-use token and hash-chained log (`web/vl/engine.js`, a port of `voicelock/`). Everything runs on the visitor's device;
the first visit downloads ~160 MB once (robust loader above). Because the secret, keys and state live in the browser, **this is a working
demonstration, not a security boundary**. Upload-first UI (no microphone needed) plus a **test mode** with a fixed phrase so the genuine
path can be tried by pre-recording the phrase on a phone. Browser test with the real models: attacker TTS reading the challenge ->
blocked (P(AI) 0.88, speaker 0.05); the enrolled speaker's real recording -> layers 1-2 pass; a different real person -> blocked; with the
phrase check relaxed for the test: verified, token executes once, tampered amount and replay denied, hash chain valid. A response takes
~3-5 s on a laptop (about 9 s in a background browser tab).

**Voice-consistency fix.** The first version of the "voice consistency" check (2 s segments vs the whole response) failed genuine users:
real microphone recordings have pauses and room tone, and a segment without speech has a meaningless embedding (a user got
consistency -0.09 with speaker match 0.71). Measured on 32 real clips with 1.6 s lead-in, a 1.4 s mid-clip pause and a 1.8 s tail: the old
check rejected 71 % of genuine clips; now the speaker layers use speech-only audio (silence trimmed) and skip segments that are mostly
pauses (`SpeakerVerifier.prepare/consistency`, `web/vl/engine.js`): 0 % rejected (minimum 0.53). Voiceprints enrolled before the fix stay valid.

### What is NOT proven (read before trusting this)

* **No voice-cloning attack was run.** There is no zero-shot cloning model in this project. The attackers above are TTS voices that
  do not sound like the enrolled speaker, so the speaker layer stopped them by itself. A real clone of the victim's voice, spoken live, is
  the threat that matters; against it only Layer 1 (and latency/liveness you add) remains, and Layer 1 is ~88 % on held-out generators, not 100 %.
  The challenge defeats replays and pre-recorded/pre-generated audio, **not** a real-time clone that reads the challenge aloud.
* The "genuine user" trials use the user's real recordings of *other* text with the transcript replaced by the expected phrase (a person
  reading arbitrary phrases was not available). Recognition of a human reading the phrase was only measured with TTS. Try it with the mic before relying on it.
* 10 speakers x 2 tiers is small; the intervals are wide. Speaker-verification data is clean read speech from a single session.
* Replay detection is the detector's training on simulated loudspeaker->mic audio plus crude acoustic heuristics, not a dedicated model (AASIST / RawNet2 were not trained).
* Not implemented: WebRTC streaming/live calls, Postgres (SQLite is used), React/Tailwind (the UI is plain HTML/JS), SIEM export, authentication of the API callers beyond one optional key, hosting (it needs a Python backend with the models; the Vercel site is the in-browser detector only).
* Active voice liveness and multi-layer deepfake detection are patented areas; the combination here (transaction-bound, session-bound challenge with a signed token) has not been checked against patents.

## Research: error guarantees on unseen AI generators (`research/PAPER.md`)

**Question.** A detector's accuracy says little about how often an AI voice from a generator it has *never seen* is accepted as real. This study
measures that with a nested **leave-one-corpus-out** protocol (9 corpora, 131 held-out generators; 45 detector fits per head), compares three ways of
choosing the threshold, and adds **domain-robust conformal calibration** (the exchangeable unit is the generator, not the clip; `voicedetect/conformal.py`).

**Results (direct-capture audio, target AI-accepted rate alpha = 10 %; MLP head; `reports/conformal_mlp.json`, logistic head replicates: `conformal_linear.json`).**

| method | held-out corpora within alpha | held-out generators within alpha | mean AI accepted per generator | genuine speech rejected |
|---|---|---|---|---|
| fixed equal-error threshold (what the app did) | 44% | 39% | 26% | 16% |
| naive split conformal | 67% | 53% | 15% | 35% |
| **domain-robust conformal (ours)** | **89%** (nominal 90 %) | 75% | 6% | 70% |

What this means, without spin:
* Domain-robust calibration restores **corpus-level** validity, but per generator it is lower (75%) because generators cluster by corpus.
* The bound is **expensive**: 70% of genuine speech is rejected at alpha = 10 %. With this detector a tight guarantee on unseen generators and a usable
  genuine-user experience do not fit together; that is why the strict limits are an **opt-in**.
* On simulated loudspeaker-to-microphone replay the detector is close to chance (AUC about 0.65), so no useful bound exists there.
* VoiceLock's default per-tier limits, audited the same way: 23%-27% of an unseen generator's AI voices accepted on average,
  22%-26% of genuine speech rejected.
* **Splice weakness:** an AI segment pasted into real speech (1 of 4 windows AI) is detected 14% of the time at 5 % false alarms; no simple aggregation rule fixes it.

**In VoiceLock** (`VOICELOCK_CONFORMAL`): `audit` (default) keeps the default limits and shows each tier's measured error on the result card; `strict` uses the calibrated limits
(browser: the "Strict authenticity limit" switch); `off` shows nothing. Live page: `/research`.

**Not done (stated in the paper):** official ASVspoof 2019 LA / In-the-Wild protocols, external baselines (AASIST, RawNet2), an adaptive white-box attacker, targeted voice clones of an enrolled
speaker. Leave-one-corpus-out describes a *new corpus*; the shipped detector is trained on all corpora.

Reproduce: `scripts/run_conformal.sh` (stage 1 `scripts/conformal_lodo.py` ~1 h, stage 2 `scripts/conformal_study.py`, export `scripts/export_research_web.py`, paper `scripts/build_paper.py`); tests: `pytest tests`.

## Tune it on your own voice

Short version for a single labelled file of your own: `python scripts/prepare_user_clip.py --ai clip.mp4`, then
`extract_features.py user --max-win 4 --aug-frac 1.0 --fresh`, `extract_replay.py user`, `train.py final` (the `user` set is up-weighted
by `USER_WEIGHT`), `export_classifier.py`. One file fixes that voice; it does not prove other voices from the same generator are fixed.
The "Teach it" flow below needs 6+ clips of each kind.

1. Open the app, go to the **Teach it** tab. Record 20+ clips (8-15 s each) of your real voice, and add 20+ AI-voice clips
   (upload files from different generators, and/or play an AI voice out loud while recording it). Clips are saved
   only on this computer in `data/user_samples/{real,ai}/`.
2. Run `.venv\Scripts\python.exe scripts	une_user.py` (add `--dry-run` to only see the report).
   It scores your clips with the current model, retrains with them (weighted x3), and reports before/after on
   clips that were held out of training. The old model is kept as `models/detector_base.joblib`.
3. Restart the app to use the tuned model.

## Ideas to improve it

More real voices from many microphones and rooms (the biggest lever), recent cloning systems (ElevenLabs, XTTS,
OpenVoice, F5-TTS), fine-tuning WavLM or training an AASIST-style head on a GPU, and a user feedback loop in the app.
