"""Embed 'speaker -> room -> microphone' degraded copies of the training windows (augment.replay).

Uses the same clip subsets/windowing as extract_features.py so clip indices line up with the clean features
(train/test splits are decided per clip id, so replayed copies can never leak across a split).
    python scripts/extract_replay.py cv libri gary asv_train unidata asv_test
"""
import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from extract_features import FEAT, WIN_S, subsample  # noqa: E402
from voicedetect.audio import SR, load_file, windows  # noqa: E402
from voicedetect.augment import replay  # noqa: E402
from voicedetect.features import Embedder  # noqa: E402

CAPS = {"gary": (500, 800), "unidata": (500, 800), "asv_train": (600, 800), "asv_test": (300, 450), "asv5": (500, 960)}   # as used for the clean features
FRAC = 0.5
CHUNK = 150


def prep(row, idx):
    y = load_file(ROOT / row["path"])
    ws = windows(y, WIN_S, hop_s=WIN_S, max_windows=2 if len(y) >= 2 * WIN_S * SR else 1)
    rng = np.random.default_rng(10_000 + idx)
    keep = [i for i in range(len(ws)) if rng.random() < FRAC]
    return [replay(ws[i], rng) for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    rows_all = [r for f in sorted((ROOT / "data" / "meta").glob("*.csv")) for r in csv.DictReader(open(f, newline=""))]
    emb = Embedder(threads=a.threads)
    pool = ThreadPoolExecutor(max_workers=3)
    for ds in a.datasets:
        cap = CAPS.get(ds, (10 ** 9, 10 ** 9))
        rows = subsample([r for r in rows_all if r["dataset"] == ds], *cap)
        cdir = FEAT / f"{ds}_replay_chunks"
        cdir.mkdir(exist_ok=True)
        print(f"[{ds}] {len(rows)} clips", flush=True)
        parts, t0 = [], time.time()
        for ci, base in enumerate(range(0, len(rows), CHUNK)):
            cp = cdir / f"{ci:04d}.npz"
            if cp.exists():
                parts.append(dict(np.load(cp)))
                continue
            sub = rows[base:base + CHUNK]
            futs = [pool.submit(prep, r, base + j) for j, r in enumerate(sub)]
            X, y, g, s, c = [], [], [], [], []
            for j, (r, f) in enumerate(zip(sub, futs)):
                ws = f.result()
                if not ws:
                    continue
                X.append(emb.embed(ws).astype(np.float16))
                y += [int(r["label"])] * len(ws); g += [r["group"]] * len(ws); s += [r["system"]] * len(ws); c += [base + j] * len(ws)
            part = dict(Xr=np.concatenate(X), y=np.array(y), group=np.array(g), system=np.array(s), clip=np.array(c))
            np.savez(cp, **part)
            parts.append(part)
            print(f"  {min(base + CHUNK, len(rows))}/{len(rows)} clips  {time.time() - t0:.0f}s", flush=True)
        np.savez(FEAT / f"{ds}_replay.npz", **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
        print(f"[{ds}] saved {sum(len(p['y']) for p in parts)} replay windows", flush=True)


if __name__ == "__main__":
    main()
