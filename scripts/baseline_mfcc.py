"""Classical baseline under the same leave-one-corpus-out protocol: MFCC (+delta) statistics -> standardised logistic regression.

Why: the WavLM detector's numbers mean little without a reference. This baseline sees the same corpora (up to 500 real + 500 AI clips each,
AI clips drawn round-robin across generators), the same silence trimming and the same 3 s window, and is evaluated on a corpus it never
trained on. It is NOT a published baseline (AASIST, RawNet2 ...); it only shows what a few hundred lines of classical DSP achieve.

    python scripts/baseline_mfcc.py
"""
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from extract_features import load_clip, subsample  # noqa: E402

DOMAINS = ["gary", "unidata", "asv_train", "cv", "libri", "asv5", "mlaad", "chatgpt", "asv_test"]


def feat(row):
    import librosa
    y = load_clip(row)[: 3 * 16000]
    if len(y) < 16000:
        y = np.pad(y, (0, 16000 - len(y)))
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=512, hop_length=160)
    d = librosa.feature.delta(m)
    return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)]).astype(np.float32)


def eer_auc(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = int(np.nanargmin(np.abs((1 - tpr) - fpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(roc_auc_score(y, s))


def main():
    X, Y, D = [], [], []
    with ThreadPoolExecutor(8) as pool:
        for d in DOMAINS:
            rows = list(csv.DictReader(open(ROOT / "data" / "meta" / f"{d}.csv", encoding="utf-8")))
            rows = subsample(rows, 500, 500)
            xs = list(pool.map(feat, rows))
            X += xs; Y += [int(r["label"]) for r in rows]; D += [d] * len(rows)
            print(d, len(rows), "clips", flush=True)
    X, Y, D = np.array(X), np.array(Y), np.array(D)
    out = {}
    for d in DOMAINS:
        tr, te = D != d, D == d
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")).fit(X[tr], Y[tr])
        s = clf.decision_function(X[te])
        if len(set(Y[te])) == 2:
            eer, auc = eer_auc(Y[te], s)
            out[d] = {"eer": eer, "auc": auc, "n_real": int((Y[te] == 0).sum()), "n_fake": int((Y[te] == 1).sum())}
            print(f"{d:10s} EER={eer:.3f} AUC={auc:.3f}", flush=True)
    (ROOT / "reports" / "baseline_mfcc.json").write_text(json.dumps({"features": "MFCC20 + delta, mean/std over 3 s", "clips_per_corpus": "<=500 real + <=500 AI",
                                                                   "protocol": "leave-one-corpus-out, clean audio", "per_corpus": out}, indent=2))


if __name__ == "__main__":
    main()
