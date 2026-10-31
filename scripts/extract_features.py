"""Embed every clip window with WavLM (clean + one augmented copy) and cache to data/features/<dataset>.npz.

    python scripts/extract_features.py gary unidata asv_train ...
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
from voicedetect.audio import load_file, windows  # noqa: E402
from voicedetect.augment import augment  # noqa: E402
from voicedetect.features import Embedder  # noqa: E402

FEAT = ROOT / "data" / "features"
WIN_S = 3.0
MAX_WIN_PER_CLIP = 2
AUG_FRAC = 0.5          # fraction of windows that also get an augmented (codec/noise/reverb) copy


# Raw corpora whose clips keep long leading/trailing silence AND where the amount of silence differs between real and fake
# (asv_test: real 1.4 s vs fake 0.6 s). The model learned "less silence = fake" from them; the app trims silence, so that
# cue vanished and real clips of this kind looked fake. They are now trimmed exactly like every other dataset.
TRIM_SETS = {"asv_train", "asv_test", "unidata"}


def load_clip(row):
    import librosa
    y = load_file(ROOT / row["path"])
    if row["dataset"] in TRIM_SETS:
        t, _ = librosa.effects.trim(y, top_db=30)
        y = np.pad(t, (int(0.1 * 16000), int(0.1 * 16000)))
    return y


def build_windows(row, seed):
    y = load_clip(row)
    # short clips -> one window; only clips well over two windows long contribute a second one
    ws = windows(y, WIN_S, hop_s=WIN_S, max_windows=MAX_WIN_PER_CLIP if len(y) >= 2 * WIN_S * 16000 else 1)
    rng = np.random.default_rng(seed)
    return ws, [augment(w, rng) for w in ws], rng.random(len(ws)) < AUG_FRAC


def subsample(rows, max_real, max_fake, seed=0):
    """Pick clips per class; fakes are drawn round-robin across generator systems so all are represented."""
    rng = np.random.default_rng(seed)
    real = [r for r in rows if r["label"] == "0"]
    fake = [r for r in rows if r["label"] == "1"]
    if len(real) > max_real:
        real = [real[i] for i in rng.choice(len(real), max_real, replace=False)]
    if len(fake) > max_fake:
        by = {}
        for r in fake:
            by.setdefault(r["system"], []).append(r)
        for v in by.values():
            rng.shuffle(v)
        out, i = [], 0
        while len(out) < max_fake:
            for v in by.values():
                if i < len(v) and len(out) < max_fake:
                    out.append(v[i])
            i += 1
        fake = out
    return sorted(real + fake, key=lambda r: r["path"])


CHUNK = 150


def process_chunk(rows, base, emb, pool):
    """Embed one chunk of clips -> dict of arrays (window level)."""
    futs = [pool.submit(build_windows, r, base + j) for j, r in enumerate(rows)]
    Xc, Xa, aug_idx, label, group, system, clip_idx = [], [], [], [], [], [], []
    n_win = 0
    for j, (r, f) in enumerate(zip(rows, futs)):
        clean, aug, mask = f.result()
        Xc.append(emb.embed(clean).astype(np.float16))
        if mask.any():
            Xa.append(emb.embed([a for a, m in zip(aug, mask) if m]).astype(np.float16))
            aug_idx += [n_win + k for k in np.flatnonzero(mask)]
        k = len(clean)
        label += [int(r["label"])] * k
        group += [r["group"]] * k
        system += [r["system"]] * k
        clip_idx += [base + j] * k
        n_win += k
    return dict(Xc=np.concatenate(Xc), Xa=np.concatenate(Xa) if Xa else np.zeros((0, *Xc[0].shape[1:]), np.float16),
                aug_idx=np.array(aug_idx, dtype=np.int64), y=np.array(label), group=np.array(group),
                system=np.array(system), clip=np.array(clip_idx))


def merge(chunks):
    out, off = {}, 0
    for c in chunks:
        c = dict(c)
        c["aug_idx"] = c["aug_idx"] + off
        off += len(c["y"])
        for k, v in c.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    global MAX_WIN_PER_CLIP, AUG_FRAC
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--max-real", type=int, default=10 ** 9)
    ap.add_argument("--max-fake", type=int, default=10 ** 9)
    ap.add_argument("--max-win", type=int, default=MAX_WIN_PER_CLIP, help="windows taken from a long clip")
    ap.add_argument("--aug-frac", type=float, default=AUG_FRAC, help="fraction of windows that also get an augmented copy")
    ap.add_argument("--fresh", action="store_true", help="ignore cached chunks for these datasets")
    a = ap.parse_args()
    MAX_WIN_PER_CLIP, AUG_FRAC = a.max_win, a.aug_frac
    FEAT.mkdir(parents=True, exist_ok=True)
    rows_all = [r for f in sorted((ROOT / "data" / "meta").glob("*.csv")) for r in csv.DictReader(open(f, newline=""))]
    emb = Embedder(threads=a.threads)
    pool = ThreadPoolExecutor(max_workers=3)
    for ds in a.datasets:
        rows = subsample([r for r in rows_all if r["dataset"] == ds], a.max_real, a.max_fake)
        print(f"[{ds}] {len(rows)} clips", flush=True)
        cdir = FEAT / f"{ds}_chunks"
        if a.fresh and cdir.exists():
            import shutil
            shutil.rmtree(cdir)
        cdir.mkdir(exist_ok=True)
        chunks, t0 = [], time.time()
        for ci, base in enumerate(range(0, len(rows), CHUNK)):
            cp = cdir / f"{ci:04d}.npz"
            if cp.exists():                                    # resume after interruption
                chunks.append(dict(np.load(cp)))
                continue
            c = process_chunk(rows[base:base + CHUNK], base, emb, pool)
            np.savez(cp, **c)
            chunks.append(c)
            done = min(base + CHUNK, len(rows))
            el = time.time() - t0
            print(f"  {done}/{len(rows)} clips  {el:.0f}s elapsed", flush=True)
        np.savez(FEAT / f"{ds}.npz", **merge(chunks))
        print(f"[{ds}] saved {sum(len(c['y']) for c in chunks)} windows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
