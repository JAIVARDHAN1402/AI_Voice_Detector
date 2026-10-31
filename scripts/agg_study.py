"""How should per-window scores be combined into one clip verdict? Evaluated on clips held out of training
(same hold-out construction as train.final): clip-level AUC / EER for several aggregators."""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402

LAYERS = [2, 3, 5, 7]


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


AGG = {
    "mean logit (current)": lambda z: z.mean(),
    "median logit": lambda z: np.median(z),
    "mean of top-50% logits": lambda z: np.sort(z)[len(z) // 2:].mean(),
    "mean of top-33% logits": lambda z: np.sort(z)[-max(1, len(z) // 3):].mean(),
    "75th percentile logit": lambda z: np.percentile(z, 75),
    "fraction of windows > 0": lambda z: (z > 0).mean(),
    "mean of clipped logits (+-3)": lambda z: np.clip(z, -3, 3).mean(),
}


def main():
    data = {n: train.load(n) for n in train.available(train.TRAIN_SETS + train.EXTRA_SETS)}
    parts = {n: train.split_holdout(data[n], n) for n in train.available(train.TRAIN_SETS) if n not in train.FAKE_ONLY}
    for n, hold_sys in train.FAKE_ONLY.items():
        if n in data:
            d = data[n]
            rng = np.random.default_rng(3)
            clips = np.unique(d["clip"])
            test_c = set(rng.choice(clips, int(len(clips) * 0.2), replace=False))
            te = np.isin(d["clip"], list(test_c)) | np.isin(d["system"], list(hold_sys))
            parts[n] = (train.take(d, ~te), train.take(d, te))
    if "asv5" in parts:
        tr, te = parts["asv5"]
        hold = np.isin(tr["system"], list(train.HOLD_ATTACKS))
        parts["asv5"] = (train.take(tr, ~hold), train.concat([te, train.take(tr, hold)]))
    trn = train.concat([p[0] for p in parts.values()])
    clf = train.fit(trn, LAYERS, 0.05)
    held = [p[1] for p in parts.values()] + [data[n] for n in train.available(train.EXTRA_SETS)]
    held = train.concat(held)
    p = train.predict(clf, held, LAYERS)
    z = lg(p)
    cat = np.where(held["replay"], "replay", np.where(held["aug"], "augmented", "clean"))
    key = np.array([f"{a}|{b}|{c}" for a, b, c in zip(held["ds"], held["clip"], cat)])
    print("windows", len(z), "clips", len(set(key)), flush=True)
    for cname in ("clean", "augmented", "replay", "ALL"):
        print(f"\n== {cname}")
        for aname, f in AGG.items():
            ys, ss = [], []
            for k in np.unique(key):
                m = key == k
                if cname != "ALL" and not k.endswith("|" + cname):
                    continue
                ys.append(int(held["y"][m][0]))
                ss.append(float(f(z[m])))
            ys, ss = np.array(ys), np.array(ss)
            if len(set(ys)) < 2:
                continue
            fpr_t = 0.10
            thr = np.quantile(ss[ys == 0], 1 - fpr_t)
            tpr = (ss[ys == 1] > thr).mean()
            print(f"  {aname:30s} clips={len(ys):5d} AUC={roc_auc_score(ys, ss):.4f}  AI caught at 10% false alarms={tpr:.3f}", flush=True)


if __name__ == "__main__":
    main()
