"""Second-stage calibrator trained on OUT-OF-DOMAIN scores.

Stage 1 scores are good at ranking but shift from corpus to corpus (the best threshold per corpus is far from 0). The stage-1 LODO runs
give, for every corpus, scores from a detector that never saw it. A small second model learns how (score, channel features) map to the
truth *in that unseen-corpus regime*. Evaluation is nested: for held-out corpus D, the calibrator is trained only on out-of-domain
scores of the OTHER corpora produced by detectors that saw neither D nor that corpus (pair fits); D is scored by the detector that
never saw D (single fit).

    python scripts/stage2_calibrator.py
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402
import conformal_study as cs  # noqa: E402

LAYERS = [2, 3, 5, 7]


def main():
    H = cs.load("mlp")
    doms = H["domains"]
    F = {}
    for D in doms:
        F[D] = train.feats(train.load(D)["X"], LAYERS)
        assert len(F[D]) == len(H["scores"][(D,)][D])
    meta = H["meta"]
    kinds = {"LR(z)": lambda: LogisticRegression(C=1.0, max_iter=500),
             "LR(z+pca32)": lambda: LogisticRegression(C=0.3, max_iter=2000),
             "GB(z+pca32)": lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.06, max_depth=4, l2_regularization=1.0, random_state=0)}
    res = {k: {"direct": {}, "all": {}} for k in ["stage1"] + list(kinds)}
    for D in doms:
        others = [r for r in doms if r != D]
        pca = PCA(32, random_state=0).fit(np.concatenate([F[r][::3] for r in others]))
        def design(Fx, z, k):
            zc = np.clip(z, -9.2, 9.2)[:, None]
            return zc if k == "LR(z)" else np.hstack([zc, pca.transform(Fx) / 30.0])
        tr_y = np.concatenate([meta[r]["y"] for r in others]); tr_ds = np.concatenate([[r] * len(meta[r]["y"]) for r in others])
        tr_z = np.concatenate([H["scores"][tuple(sorted((D, r)))][r] for r in others]); tr_F = np.concatenate([F[r] for r in others])
        w = train.cell_weights(tr_ds, tr_y)
        te_z = H["scores"][(D,)][D]
        for scope, conds in (("direct", ("clean", "augmented")), ("all", None)):
            y, s = cs.clip_table(meta[D], te_z, conds)["y"], cs.clip_table(meta[D], te_z, conds)["s"]
            res["stage1"][scope][D] = (float(((s > 0).astype(int) == y).mean()), float(roc_auc_score(y, s)) if len(set(y)) == 2 else None)
        for k, mk in kinds.items():
            m = mk()
            Xtr = design(tr_F, tr_z, k)
            m.fit(Xtr, tr_y, sample_weight=w)
            p = m.predict_proba(design(F[D], te_z, k))[:, 1]
            zz = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            for scope, conds in (("direct", ("clean", "augmented")), ("all", None)):
                t = cs.clip_table(meta[D], zz, conds)
                res[k][scope][D] = (float(((t["s"] > 0).astype(int) == t["y"]).mean()), float(roc_auc_score(t["y"], t["s"])) if len(set(t["y"])) == 2 else None)
        print("done", D, flush=True)
    for k, r in res.items():
        for scope in ("direct", "all"):
            a = np.mean([v[0] for v in r[scope].values()]); u = np.mean([v[1] for v in r[scope].values() if v[1] is not None])
            print(f"{k:12s} {scope:7s} macro-acc {100 * a:5.1f}%  macro-AUC {u:.3f}  per corpus {{{', '.join(f'{d}:{100 * v[0]:.0f}' for d, v in r[scope].items())}}}", flush=True)


if __name__ == "__main__":
    main()
