"""Build the 'libri' dataset: LibriSpeech readers (train-clean-100, never test-clean) vs. neural TTS of the same text.

Each parquet row group is one speaker, so we sample whole row groups from two shards for speaker diversity.
The held-out check in ood_check.py uses test-clean speakers, which are disjoint from train-clean-100.
"""
import asyncio
import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_cv import FIELDS, synth_all, trim  # noqa: E402
from voicedetect.audio import SR, load_bytes  # noqa: E402

OUT = ROOT / "data" / "processed" / "libri"
META = ROOT / "data" / "meta" / "libri.csv"
SHARDS = {0: 10, 3: 10}          # shard -> number of speakers (row groups) to read
PER_SPEAKER = 16
N_FAKE = 170


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    rng = np.random.default_rng(7)
    OUT.mkdir(parents=True, exist_ok=True)
    real_json = ROOT / "data" / "libri_real.json"
    if real_json.exists():
        saved = json.loads(real_json.read_text())
        meta, sentences = saved["meta"], saved["sentences"]
        print(f"reusing {len(meta)} real clips", flush=True)
    else:
        fs = HfFileSystem()
        meta, sentences = [], []
        for shard, n_spk in SHARDS.items():
            remote = f"datasets/openslr/librispeech_asr/all/train.clean.100/{shard:04d}.parquet"
            with fs.open(remote, "rb", block_size=4 * 1024 * 1024) as fh:
                pf = pq.ParquetFile(fh)
                for g in rng.choice(pf.metadata.num_row_groups, n_spk, replace=False):
                    t = pf.read_row_group(int(g), columns=["audio", "text", "speaker_id"])
                    rows = list(zip(t.column("audio").to_pylist(), t.column("text").to_pylist(),
                                    t.column("speaker_id").to_pylist()))
                    rng.shuffle(rows)
                    k = 0
                    for a, text, spk in rows:
                        if k >= PER_SPEAKER:
                            break
                        y = trim(load_bytes(a["bytes"], ".flac"))
                        if not (3.0 <= len(y) / SR <= 12.0):
                            continue
                        name = f"real_{len(meta):04d}"
                        sf.write(OUT / f"{name}.flac", y, SR, subtype="PCM_16", format="FLAC")
                        meta.append(dict(path=f"data/processed/libri/{name}.flac", label=0, dataset="libri",
                                         group=f"libri_{spk}", system="real"))
                        sentences.append(text.capitalize())
                        k += 1
                    print(f"  shard {shard} speaker {spk}: {len(meta)} real clips so far", flush=True)
        real_json.write_text(json.dumps({"meta": meta, "sentences": sentences}))

    import edge_tts
    voices = [v["ShortName"] for v in asyncio.run(edge_tts.list_voices()) if v["Locale"].startswith("en-")]
    pick = rng.choice(len(sentences), N_FAKE, replace=False)
    jobs = [(" ".join(sentences[i].split()[:30]), voices[int(rng.integers(len(voices)))]) for i in pick]
    cache = ROOT / "data" / "libri_tts_cache"
    cache.mkdir(exist_ok=True)
    print(f"synthesising {len(jobs)} sentences ...", flush=True)
    asyncio.run(synth_all(jobs, cache))
    nf = 0
    for i, (text, voice) in enumerate(jobs):
        f = cache / f"{i:04d}.mp3"
        if not f.exists():
            continue
        try:
            y = trim(load_bytes(f.read_bytes(), ".mp3"))
        except Exception:  # noqa: BLE001
            continue
        if len(y) / SR < 1.5:
            continue
        name = f"tts_{i:04d}"
        sf.write(OUT / f"{name}.flac", y, SR, subtype="PCM_16", format="FLAC")
        meta.append(dict(path=f"data/processed/libri/{name}.flac", label=1, dataset="libri",
                         group=voice, system="edge:" + "-".join(voice.split("-")[:2])))
        nf += 1
    with open(META, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in meta:
            w.writerow({k: m.get(k, "") for k in FIELDS})
    print(f"fake clips: {nf}\nDONE", flush=True)


if __name__ == "__main__":
    main()
