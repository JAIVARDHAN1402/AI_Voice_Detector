"""Train / evaluate the detector on cached WavLM features.

    python scripts/train.py explore   # leave-one-dataset-out study: which layers / regularisation generalise?
    python scripts/train.py final     # honest hold-out evaluation, then refit on everything -> models/detector.joblib
"""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TRAIN_SETS = ["gary", "unidata", "asv_train", "cv", "libri"]
EXTRA_SETS = ["asv_test"]          # unseen speakers + unseen attack systems (A07-A19)


def load(name):
    """-> dict with one entry per *sample*: X (clean and augmented copies stacked), y, group, ds, aug flag."""
    z = np.load(FEAT / f"{name}.npz")
    ai = z["aug_idx"]
    y, grp = z["y"], np.array([re.sub(r"_p2$", "", g) for g in z["group"]])
    return {
        "X": np.concatenate([z["Xc"], z["Xa"]]),
        "y": np.concatenate([y, y[ai]]),
        "group": np.concatenate([grp, grp[ai]]),
        "system": np.concatenate([z["system"], z["system"][ai]]),
        "clip": np.concatenate([z["clip"], z["clip"][ai]]),
        "aug": np.concatenate([np.zeros(len(y), bool), np.ones(len(ai), bool)]),
        "ds": np.array([name] * (len(y) + len(ai))),
    }


def concat(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def take(d, m):
    return {k: v[m] for k, v in d.items()}


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def metrics(y, p, thr=0.5):
    if len(np.unique(y)) < 2:
        return {}
    pred = (p >= thr).astype(int)
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, p)), "eer": eer(y, p)[0],
            "acc": float((pred == y).mean()), "real_acc": float((pred[y == 0] == 0).mean()),
            "fake_acc": float((pred[y == 1] == 1).mean())}


def cell_weights(ds, y):
    """Each (dataset, label) cell gets sqrt-balanced total weight so no corpus or class dominates."""
    w = np.zeros(len(y), np.float32)
    for d in np.unique(ds):
        for lab in (0, 1):
            m = (ds == d) & (y == lab)
            if m.any():
                w[m] = 1.0 / np.sqrt(m.sum())
    return w * len(y) / w.sum()


def feats(X, layers):
    if layers == "mean":
        return X.astype(np.float32).mean(axis=1)
    return X[:, layers, :].reshape(len(X), -1).astype(np.float32)


def make_clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def fit(d, layers, C):
    clf = make_clf(C)
    clf.fit(feats(d["X"], layers), d["y"], logisticregression__sample_weight=cell_weights(d["ds"], d["y"]))
    return clf


def predict(clf, d, layers):
    return clf.predict_proba(feats(d["X"], layers))[:, 1]


def available(names):
    return [n for n in names if (FEAT / f"{n}.npz").exists()]


def explore():
    data = {n: load(n) for n in available(TRAIN_SETS + EXTRA_SETS)}
    for n, d in data.items():
        print(f"{n}: {len(d['y'])} samples, {int(d['y'].sum())} fake, {len(np.unique(d['group']))} groups", flush=True)
    configs = {"mean_all": "mean", "L2": [2], "L4": [4], "L6": [6], "L8": [8], "L3+5+7": [3, 5, 7], "L0-8 even": [0, 2, 4, 6, 8]}
    for held in data:
        train = concat([d for n, d in data.items() if n != held and n in TRAIN_SETS])
        for cname, layers in configs.items():
            for C in (0.01, 0.1):
                clf = fit(train, layers, C)
                for aug in (False, True):
                    m = data[held]["aug"] == aug
                    if not m.any():
                        continue
                    r = metrics(data[held]["y"][m], predict(clf, take(data[held], m), layers))
                    if r:
                        print(f"held-out={held:9s} {cname:10s} C={C:<5} {'aug  ' if aug else 'clean'} AUC={r['auc']:.3f} "
                              f"EER={r['eer']:.3f} acc={r['acc']:.3f} real={r['real_acc']:.2f} fake={r['fake_acc']:.2f}", flush=True)


def split_holdout(d, name, frac=0.2, seed=0):
    """Hold out whole speakers / recordings (never windows) so voices are not memorised."""
    rng = np.random.default_rng(seed)
    te = np.zeros(len(d["y"]), bool)
    if name == "gary":     # fake 'groups' are TTS vendors, so hold out real recordings + a random 20% of fake clips
        real_groups = np.unique(d["group"][d["y"] == 0])
        test_g = set(rng.choice(real_groups, max(1, len(real_groups) // 5), replace=False))
        te[(d["y"] == 0) & np.isin(d["group"], list(test_g))] = True
        fake_clips = np.unique(d["clip"][d["y"] == 1])
        test_c = set(rng.choice(fake_clips, int(len(fake_clips) * frac), replace=False))
        te[(d["y"] == 1) & np.isin(d["clip"], list(test_c))] = True
    else:
        # real: hold out speakers; fake: hold out clips (their group is the attack system, not a person)
        rg = np.unique(d["group"][d["y"] == 0])
        test_g = set(rng.choice(rg, max(1, int(round(len(rg) * frac))), replace=False))
        te[(d["y"] == 0) & np.isin(d["group"], list(test_g))] = True
        fake_clips = np.unique(d["clip"][d["y"] == 1])
        test_c = set(rng.choice(fake_clips, int(len(fake_clips) * frac), replace=False))
        te[(d["y"] == 1) & np.isin(d["clip"], list(test_c))] = True
    return take(d, ~te), take(d, te)


def final(layers, C):
    data = {n: load(n) for n in available(TRAIN_SETS + EXTRA_SETS)}
    parts = {n: split_holdout(data[n], n) for n in available(TRAIN_SETS)}
    train = concat([p[0] for p in parts.values()])
    clf = fit(train, layers, C)

    report = {"layers": layers, "C": C, "train_samples": int(len(train["y"])), "eval": {}}
    pooled_p, pooled_y = [], []
    evals = {f"{n} (held-out speakers/recordings)": p[1] for n, p in parts.items()}
    for n in available(EXTRA_SETS):
        evals[f"{n} (unseen attacks + speakers)"] = data[n]
    for name, d in evals.items():
        for aug in (False, True):
            m = d["aug"] == aug
            if not m.any():
                continue
            p = predict(clf, take(d, m), layers)
            r = metrics(d["y"][m], p)
            if r:
                report["eval"][f"{name} / {'augmented' if aug else 'clean'}"] = r
                pooled_p.append(p); pooled_y.append(d["y"][m])
    pp, yy = np.concatenate(pooled_p), np.concatenate(pooled_y)
    _, thr = eer(yy, pp)
    report["pooled"] = metrics(yy, pp, thr) | {"threshold": thr}
    # per-attack-system detection rate on unseen ASVspoof attacks
    if "asv_test" in data:
        d = data["asv_test"]
        p = predict(clf, d, layers)
        report["asv_test_per_system_fake_detection"] = {
            s: float((p[(d["system"] == s)] >= thr).mean()) for s in sorted(set(d["system"])) if s != "real"}

    # refit on every training sample (including the held-out parts and the unseen-attack set) for deployment
    full = concat([data[n] for n in data])
    clf_full = fit(full, layers, C)
    return clf_full, report, thr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["explore", "final"])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--C", type=float, default=0.05)
    a = ap.parse_args()
    if a.mode == "explore":
        explore()
    else:
        clf, report, thr = final(a.layers, a.C)
        REPORTS.mkdir(exist_ok=True)
        MODELS.mkdir(exist_ok=True)
        (REPORTS / "evaluation.json").write_text(json.dumps(report, indent=2))
        joblib.dump({"clf": clf, "layers": a.layers, "threshold": float(thr), "unsure_margin": 0.15,
                     "meta": {"backbone": "microsoft/wavlm-base-plus (8 layers)", "datasets": TRAIN_SETS + EXTRA_SETS,
                              "pooled_eer": report["pooled"]["eer"]}}, MODELS / "detector.joblib")
        print(json.dumps(report, indent=2))
