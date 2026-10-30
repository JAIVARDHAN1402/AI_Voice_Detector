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
| Verdict policy on held-out windows: AI called "real" / real called "AI" | 8 % / 8 % |
| Real voices called real / AI called AI | 86 % / 87 % |
| Independent check: unseen LibriSpeech readers correct / wrong; Windows + Google TTS | 88 % / 10 %; 100 % |

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
web/model/               wavlm_pooled.<hash>.onnx (69 MB: int8 matmuls + fp16-stored convs), head.<hash>.bin (6 MB MLP head), classifier.json
web/ort/                 vendored onnxruntime-web 1.30.0 (MIT)
web/vercel.json          cross-origin-isolation headers (multi-threaded WASM) + caching
```

Rebuild after retraining: `export_onnx.py` -> `quantize_onnx.py` -> `export_classifier.py` (writes the hashed model name and the
folded classifier). Test locally with `python scripts/dev_web.py` (same headers as Vercel) and deploy with
`cd web && npx vercel --prod`.

Checks: the ONNX export matches PyTorch to ~2e-6; int8 quantisation shifts window logits by 0.6 (std) with 0 flipped
decisions on 80 windows; the browser pipeline gave the same verdicts as the Python detector on 10/10 test files and identical
clip durations (P(AI) within a few points; ~0.4 s per 3 s segment on a laptop).
The "Teach it" tab and `tune_user.py` are part of the local Flask app only.

## Tune it on your own voice

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
