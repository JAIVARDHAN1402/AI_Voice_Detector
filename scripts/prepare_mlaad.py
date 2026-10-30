"""Build the 'mlaad' dataset: English clips from 64 recent TTS / voice-cloning systems (MLAAD-tiny, CC BY-NC 4.0).

Fake only (real speech comes from the other datasets). `system` = the generator name, so evaluation can hold whole
generators out. ~24 random clips per system, silence-trimmed, 16 kHz FLAC.
"""
import csv
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_cv import FIELDS, trim  # noqa: E402
from voicedetect.audio import SR, load_file  # noqa: E402

REPO = "mueller91/MLAAD-tiny"
RAW = ROOT / "data" / "raw" / "mlaad"
OUT = ROOT / "data" / "processed" / "mlaad"
META = ROOT / "data" / "meta" / "mlaad.csv"
PER_SYSTEM = 24


def main():
    import re
    import time
    from huggingface_hub import HfApi, hf_hub_download
    rng = np.random.default_rng(11)
    files = [s.rfilename for s in HfApi().dataset_info(REPO, files_metadata=False).siblings]
    by = {}
    for f in files:
        p = f.split("/")
        if len(p) >= 4 and p[0] == "fake" and p[1] == "en" and f.lower().endswith((".wav", ".flac", ".mp3")):
            by.setdefault(p[2], []).append(f)
    chosen = []
    for sysname, fs in sorted(by.items()):
        fs = sorted(fs)
        chosen += [(sysname, fs[i]) for i in rng.choice(len(fs), min(PER_SYSTEM, len(fs)), replace=False)]
    print(f"{len(by)} systems, {len(chosen)} clips to fetch", flush=True)

    def get(item):
        for attempt in range(4):
            try:
                return item, hf_hub_download(REPO, item[1], repo_type="dataset", local_dir=RAW)
            except Exception:  # noqa: BLE001
                time.sleep(2 + attempt)
        return item, None

    OUT.mkdir(parents=True, exist_ok=True)
    meta, done = [], 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (sysname, rel), local in pool.map(get, chosen):
            done += 1
            if local is None:
                continue
            try:
                y = trim(load_file(local))
            except Exception:  # noqa: BLE001
                continue
            if not (1.5 <= len(y) / SR <= 14):
                continue
            name = f"{len(meta):05d}"
            sf.write(OUT / f"{name}.flac", y[: 12 * SR], SR, subtype="PCM_16", format="FLAC")
            sid = re.sub(r"[^A-Za-z0-9._-]+", "_", sysname)
            meta.append(dict(path=f"data/processed/mlaad/{name}.flac", label=1, dataset="mlaad", group=sid, system=sid))
            if done % 100 == 0:
                print(f"  {done}/{len(chosen)} fetched, {len(meta)} kept", flush=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    with open(META, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: m.get(k, "") for k in FIELDS} for m in meta)
    print(f"DONE {len(meta)} clips from {len(set(m['system'] for m in meta))} systems", flush=True)


if __name__ == "__main__":
    main()
