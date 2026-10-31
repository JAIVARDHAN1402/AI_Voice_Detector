"""Stage 1 of the conformal study: out-of-domain window scores for every (test domain, calibration domain) pair.

For the domain-robust calibration to be honest, the scores used to *calibrate* a threshold for a held-out domain D must
come from detectors that never saw the calibration domain either, and the detector that is finally *tested* on D must not
have seen D. That is a nested leave-one-domain-out design:

    single[D]      : detector trained on every domain except D                    -> scores of D        (the test scores)
    pair[(a, b)]   : detector trained on every domain except a and b              -> scores of a and b  (calibration
                     scores of a when D = b, and of b when D = a; the detector saw neither)

9 domains -> 9 + 36 = 45 fits. The clip-level analysis (scripts/conformal_study.py) only reads the file written here.

    python scripts/conformal_lodo.py [--head mlp|linear] [--jobs 4]
"""
import argparse
import itertools
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402

LAYERS = [2, 3, 5, 7]
DOMAINS = ["gary", "unidata", "asv_train", "cv", "libri", "asv5", "mlaad", "chatgpt", "asv_test"]
ALWAYS_TRAIN = ["user"]          # a handful of the author's own clips: never held out, always in the training set
OUT = ROOT / "data" / "conformal"


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).astype(np.float32)


def fit_and_score(head, excluded, score_sets):
    train.HEAD = head
    names = [n for n in DOMAINS + ALWAYS_TRAIN if n not in excluded and (train.FEAT / f"{n}.npz").exists()]
    data = {n: train.load(n) for n in names + [s for s in score_sets if s not in names]}
    t = time.time()
    clf = train.fit(train.concat([data[n] for n in names]), LAYERS, 0.05 if head == "mlp" else 0.01)
    out = {s: logit(train.predict(clf, data[s], LAYERS)) for s in score_sets}
    print(f"[{head}] excluded={sorted(excluded)} fit+score {time.time() - t:.0f}s", flush=True)
    return tuple(sorted(excluded)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="mlp", choices=["mlp", "linear"])
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = {}
    for n in DOMAINS:
        d = train.load(n)
        meta[n] = {k: d[k] for k in ("y", "system", "clip", "aug", "replay")}
    jobs = [((d,), [d]) for d in DOMAINS] + [((x, y), [x, y]) for x, y in itertools.combinations(DOMAINS, 2)]
    res = Parallel(n_jobs=a.jobs, verbose=0)(delayed(fit_and_score)(a.head, set(ex), sc) for ex, sc in jobs)
    scores = dict(res)
    with open(OUT / f"lodo_scores_{a.head}.pkl", "wb") as f:
        pickle.dump({"meta": meta, "scores": scores, "domains": DOMAINS, "layers": LAYERS, "head": a.head}, f)
    print("saved", OUT / f"lodo_scores_{a.head}.pkl")


if __name__ == "__main__":
    main()
