"""Leave-one-dataset-out comparison of classifier heads on the cached WavLM features.

For each held-out dataset H: fit on every other dataset (asv_test is always train-only) and score H.
Reports AUC and the AI detection rate at a 10 % false-alarm rate on real voices (TPR@FPR10) - threshold-free,
so it measures how well real and AI are *ranked* on a domain the head never saw.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

HELD = ["gary", "cv", "libri", "asv5", "asv_train"]
CONFIGS = {
    "LR C=0.005  L2,3,5,7": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.005, max_iter=3000)), [2, 3, 5, 7]),
    "LR C=0.05   L2,3,5,7 (current)": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)), [2, 3, 5, 7]),
    "LR C=0.5    L2,3,5,7": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)), [2, 3, 5, 7]),
    "LR C=0.02   L0-8 (all)": (lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.02, max_iter=3000)), list(range(9))),
    "MLP 256     L2,3,5,7": (lambda: make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=0.05, early_stopping=True,
                                                                                    n_iter_no_change=6, max_iter=80, random_state=0)), [2, 3, 5, 7]),
}


def tpr_at_fpr(y, s, fpr_t=0.10):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_t, fpr, tpr))


def fit(make, layers, d):
    clf = make()
    w = train.cell_weights(d["ds"], d["y"])
    last = clf.steps[-1][0]
    clf.fit(train.feats(d["X"], layers), d["y"], **{f"{last}__sample_weight": w})
    return clf


def main():
    data = {n: train.load(n) for n in train.available(train.TRAIN_SETS + train.EXTRA_SETS)}
    print({n: len(d["y"]) for n, d in data.items()}, flush=True)
    res = {c: {} for c in CONFIGS}
    for h in HELD:
        tr = train.concat([d for n, d in data.items() if n != h])
        for cname, (make, layers) in CONFIGS.items():
            t0 = time.time()
            clf = fit(make, layers, tr)
            p = clf.predict_proba(train.feats(data[h]["X"], layers))[:, 1]
            y = data[h]["y"]
            res[cname][h] = (roc_auc_score(y, p), tpr_at_fpr(y, p))
            print(f"held-out={h:9s} {cname:32s} AUC={res[cname][h][0]:.3f} TPR@FPR10={res[cname][h][1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
    print("\nMEAN over held-out datasets:")
    for cname, r in res.items():
        print(f"  {cname:32s} AUC={np.mean([v[0] for v in r.values()]):.3f}  TPR@FPR10={np.mean([v[1] for v in r.values()]):.2f}")


if __name__ == "__main__":
    main()
