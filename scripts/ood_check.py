"""Out-of-distribution sanity check with data that never touched training or threshold selection.

Real : LibriSpeech test-clean readers (audiobook volunteers, different corpus/speakers/microphones).
Fake : Windows SAPI voices (David, Zira, Hazel) and Google TTS (gTTS, several accents) reading the same text.
Runs the *full* Detector pipeline (silence trim, windows, aggregation, thresholds) on each file.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicedetect.audio import SR, load_bytes  # noqa: E402

D = ROOT / "data" / "ood"
N_REAL, N_SAPI, N_GTTS = 48, 30, 30


def fetch_libri(rng):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    out, seen = [], {}
    with fs.open("datasets/openslr/librispeech_asr/all/test.clean/0000.parquet", "rb", block_size=4 * 1024 * 1024) as fh:
        pf = pq.ParquetFile(fh)
        print("librispeech row groups:", pf.metadata.num_row_groups, flush=True)
        for g in rng.permutation(pf.metadata.num_row_groups)[:14]:
            t = pf.read_row_group(int(g), columns=["audio", "text", "speaker_id"])
            for a, text, spk in zip(t.column("audio").to_pylist(), t.column("text").to_pylist(), t.column("speaker_id").to_pylist()):
                y = load_bytes(a["bytes"], ".flac")
                if seen.get(spk, 0) >= 2 or not (4 <= len(y) / SR <= 14):
                    continue
                seen[spk] = seen.get(spk, 0) + 1
                out.append((y, text.capitalize()))
            print(f"  row group {g}: {len(out)} clips / {len(seen)} speakers", flush=True)
            if len(out) >= N_REAL:
                break
    return out[:N_REAL]


def sapi(text, voice, path):
    text = text.replace("'", "''")
    ps = (f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SelectVoice('{voice}'); $s.SetOutputToWaveFile('{path}'); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)


def main():
    rng = np.random.default_rng(123)
    (D / "real").mkdir(parents=True, exist_ok=True)
    (D / "sapi").mkdir(exist_ok=True)
    (D / "gtts").mkdir(exist_ok=True)
    manifest = []
    texts = []
    if not list((D / "real").glob("*.flac")):
        for i, (y, text) in enumerate(fetch_libri(rng)):
            sf.write(D / "real" / f"libri_{i:03d}.flac", y, SR, subtype="PCM_16")
            texts.append(text)
        (D / "texts.json").write_text(json.dumps(texts))
    texts = json.loads((D / "texts.json").read_text())
    words = lambda t: " ".join(t.split()[:28])  # noqa: E731

    voices = ["Microsoft David Desktop", "Microsoft Zira Desktop", "Microsoft Hazel Desktop"]
    for i in range(N_SAPI):
        p = D / "sapi" / f"sapi_{i:03d}.wav"
        if not p.exists():
            sapi(words(texts[i % len(texts)]), voices[i % 3], str(p))
    tlds = ["com", "co.uk", "com.au", "co.in", "ca"]
    for i in range(N_GTTS):
        p = D / "gtts" / f"gtts_{i:03d}.mp3"
        if not p.exists():
            from gtts import gTTS   # optional: only needed to (re)generate the test files
            gTTS(words(texts[(N_SAPI + i) % len(texts)]), lang="en", tld=tlds[i % 5]).save(str(p))
    print("test files ready", flush=True)

    from voicedetect.detector import Detector
    det = Detector()
    groups = {"real (LibriSpeech)": (sorted((D / "real").glob("*.flac")), 0),
              "fake (Windows SAPI)": (sorted((D / "sapi").glob("*.wav")), 1),
              "fake (Google TTS)": (sorted((D / "gtts").glob("*.mp3")), 1)}
    report = {}
    for name, (files, label) in groups.items():
        res = [det.predict_file(f) for f in files]
        v = [r["verdict"] for r in res]
        p = np.array([r["prob_ai"] for r in res])
        correct = "real" if label == 0 else "ai"
        report[name] = {"n": len(files), "correct": sum(x == correct for x in v) / len(v),
                        "unsure": sum(x == "unsure" for x in v) / len(v),
                        "wrong": sum(x not in (correct, "unsure") for x in v) / len(v),
                        "median_prob_ai": float(np.median(p))}
        print(f"{name:22s} n={len(files):3d} correct={report[name]['correct']:.0%} unsure={report[name]['unsure']:.0%} "
              f"wrong={report[name]['wrong']:.0%} median P(AI)={report[name]['median_prob_ai']:.2f}", flush=True)
    (ROOT / "reports" / "ood_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
