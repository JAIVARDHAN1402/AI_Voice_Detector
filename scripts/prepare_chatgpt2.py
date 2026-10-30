"""Build the 'chatgpt2' dataset: long, chat-style answers spoken by gpt-4o-mini-tts (LGB666/SageLM-gpt-4o-mini-tts).

Fake only. These are streamed 24 kHz WAV answers to instructions (much closer to what ChatGPT voice says than the short
AdvBench prompts). A handful of the parquet shards is downloaded, ~50 clips per shard are kept and 2 random 3-10 s segments
are cut from each. `system` = shard, `group` = original file name.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_cv import FIELDS, trim  # noqa: E402
from voicedetect.audio import SR, load_bytes  # noqa: E402

REPO = "LGB666/SageLM-gpt-4o-mini-tts"
RAW = ROOT / "data" / "raw" / "chatgpt2"
OUT = ROOT / "data" / "processed" / "chatgpt2"
META = ROOT / "data" / "meta" / "chatgpt2.csv"
SHARDS = [0, 14, 28, 42, 56]
CLIPS_PER_SHARD = 50
SEGS_PER_CLIP = 2


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    rng = np.random.default_rng(31)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    meta = []
    for sh in SHARDS:
        name = f"gpt-4o-mini-tts_{sh:04d}.parquet"
        print("download", name, flush=True)
        local = hf_hub_download(REPO, name, repo_type="dataset", local_dir=RAW)
        pf = pq.ParquetFile(local)
        n = 0
        for g in range(pf.metadata.num_row_groups):
            tab = pf.read_row_group(g)
            fn, au = tab.column("file_name").to_pylist(), tab.column("audio_bytes").to_pylist()
            for i in rng.choice(len(fn), min(CLIPS_PER_SHARD, len(fn)), replace=False):
                try:
                    y = load_bytes(au[i], ".wav")
                except Exception:  # noqa: BLE001
                    continue
                y = trim(y)
                if len(y) < 4 * SR:
                    continue
                for _ in range(SEGS_PER_CLIP):
                    dur = int(rng.uniform(3, 10) * SR)
                    s = int(rng.integers(0, max(1, len(y) - dur)))
                    seg = trim(y[s:s + dur])
                    if len(seg) < 2 * SR:
                        continue
                    k = f"{len(meta):05d}"
                    sf.write(OUT / f"{k}.flac", seg, SR, subtype="PCM_16", format="FLAC")
                    meta.append(dict(path=f"data/processed/chatgpt2/{k}.flac", label=1, dataset="chatgpt2",
                                     group=re.sub(r"[^A-Za-z0-9._-]+", "_", fn[i]), system=f"gpt4o_mini_tts_sage_{sh:04d}"))
                    n += 1
            del tab
        print(f"  shard {sh}: {n} segments", flush=True)
        del pf
        Path(local).unlink()
    META.parent.mkdir(parents=True, exist_ok=True)
    with open(META, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: m.get(k, "") for k in FIELDS} for m in meta)
    print(f"DONE {len(meta)} segments", flush=True)


if __name__ == "__main__":
    main()
