"""Which training procedure generalises best to a corpus the detector has never seen?

Leave-one-corpus-out over the cached WavLM features (layers 2,3,5,7 -> 6144 features), same clip-level score (mean window logit),
same metric as the baseline: macro accuracy over the held-out corpora at the model's own 0.5 point (no threshold tuning on the
held-out corpus). Every variant keeps the deployed architecture family (one hidden ReLU layer on standardised features), so
the winner can be exported to the web build unchanged (an average of N such MLPs is again one wider MLP).

    python scripts/dg_experiments.py [name ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402
import conformal_study as cs  # noqa: E402

torch.set_num_threads(16)
LAYERS = [2, 3, 5, 7]
EVAL = ["gary", "unidata", "asv_train", "cv", "libri", "asv5", "mlaad", "chatgpt", "asv_test"]
BASE_TRAIN = EVAL + ["user"]
OUT = ROOT / "reports" / "dg_experiments.json"

_DATA = {}


def data(name):
    if name not in _DATA:
        d = train.load(name)
        d["F"] = train.feats(d["X"], LAYERS)
        del d["X"]
        _DATA[name] = d
    return _DATA[name]


def stack(names):
    parts = [data(n) for n in names]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class Net(torch.nn.Module):
    def __init__(self, d_in, hidden, drop):
        super().__init__()
        self.l1, self.l2, self.drop = torch.nn.Linear(d_in, hidden), torch.nn.Linear(hidden, 1), torch.nn.Dropout(drop)

    def forward(self, x):
        return self.l2(self.drop(torch.relu(self.l1(x)))).squeeze(-1)


def fit_predict(tr, test_sets, cfg, seed):
    """Train one network on `tr`, return logits for each array in test_sets."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    mu, sd = tr["F"].mean(0), tr["F"].std(0) + 1e-6
    X = torch.from_numpy((tr["F"] - mu) / sd)
    y = torch.from_numpy(tr["y"].astype(np.float32))
    w = train.cell_weights(tr["ds"], tr["y"]) * np.where(tr["ds"] == "user", train.USER_WEIGHT, 1.0)
    w = torch.from_numpy((w / w.mean()).astype(np.float32))
    cells = np.array([f"{a}|{b}" for a, b in zip(tr["ds"], tr["y"])])
    cell_ids = torch.from_numpy(np.unique(cells, return_inverse=True)[1])
    n_cells = int(cell_ids.max()) + 1
    q = torch.ones(n_cells) / n_cells
    net = Net(X.shape[1], cfg["hidden"], cfg["drop"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n = len(X)
    for ep in range(cfg["epochs"]):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, cfg["batch"]):
            b = perm[i:i + cfg["batch"]]
            xb = X[b]
            if cfg["noise"]:
                xb = xb + cfg["noise"] * torch.randn_like(xb)
            if cfg["mixup"]:                                   # mix windows of the SAME label (keeps the label, blends channels)
                lam = torch.distributions.Beta(cfg["mixup"], cfg["mixup"]).sample((len(b), 1))
                same = {}
                idx = torch.arange(len(b))
                for lab in (0, 1):
                    m = (y[b] == lab).nonzero().squeeze(-1)
                    if len(m) > 1:
                        idx[m] = m[torch.randperm(len(m))]
                xb = lam * xb + (1 - lam) * xb[idx]
            loss_i = torch.nn.functional.binary_cross_entropy_with_logits(net(xb), y[b], reduction="none") * w[b]
            if cfg["dro"]:                                     # group DRO over (corpus, label) cells: emphasise the worst cells
                cb = cell_ids[b]
                gl = torch.zeros(n_cells).index_add(0, cb, loss_i.detach()) / torch.bincount(cb, minlength=n_cells).clamp(min=1)
                q = q * torch.exp(cfg["dro"] * gl); q = q / q.sum()
                loss = (loss_i * (q[cb] * n_cells)).mean()
            else:
                loss = loss_i.mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval()
    with torch.no_grad():
        return [net(torch.from_numpy((t["F"] - mu) / sd)).numpy() for t in test_sets], (net, mu, sd)


def clip_scores(d, z, conds):
    t = cs.clip_table({k: d[k] for k in ("clip", "aug", "replay", "y", "system")}, z, conds)
    return t["y"], t["s"]


def lodo(cfg, extra_train=(), seeds=(0,)):
    res = {"direct": {}, "all": {}}
    for held in EVAL:
        tr = stack([n for n in BASE_TRAIN + list(extra_train) if n != held])
        te = data(held)
        zs = []
        for s in seeds:
            (z,), _ = fit_predict(tr, [te], cfg, s)
            zs.append(z)
        z = np.mean(zs, axis=0)                                # averaging logits of N nets == one wider net
        for scope, conds in (("direct", ("clean", "augmented")), ("all", None)):
            y, s_ = clip_scores(te, z, conds)
            acc = float(((s_ > 0).astype(int) == y).mean())
            auc = float(roc_auc_score(y, s_)) if len(set(y)) == 2 else None
            res[scope][held] = {"acc": acc, "auc": auc}
    out = {}
    for scope, r in res.items():
        out[scope] = {"macro_acc": float(np.mean([v["acc"] for v in r.values()])),
                      "macro_auc": float(np.mean([v["auc"] for v in r.values() if v["auc"] is not None])),
                      "worst_acc": float(min(v["acc"] for v in r.values())), "per_corpus": r}
    return out


BASE = dict(hidden=256, drop=0.0, wd=0.05, lr=1e-3, epochs=20, batch=256, noise=0.0, mixup=0.0, dro=0.0)
EXPERIMENTS = {
    "e0_baseline_torch": (BASE, (), (0,)),
    "e1_drop_noise": ({**BASE, "drop": 0.3, "noise": 0.3, "wd": 0.01}, (), (0,)),
    "e2_dro": ({**BASE, "dro": 0.01}, (), (0,)),
    "e3_chatgpt2": (BASE, ("chatgpt2",), (0,)),
    "e4_mixup": ({**BASE, "mixup": 0.4}, (), (0,)),
    "e7_combo": ({**BASE, "dro": 0.01}, ("chatgpt2",), (0, 1, 2)),
    "e5_wide": ({**BASE, "hidden": 1024, "drop": 0.5, "noise": 0.3, "wd": 0.01}, (), (0,)),
    "e6_short": ({**BASE, "drop": 0.3, "noise": 0.5, "wd": 0.01, "epochs": 6}, (), (0,)),
}


def main():
    names = sys.argv[1:] or list(EXPERIMENTS)
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        cfg, extra, seeds = EXPERIMENTS[name]
        t = time.time()
        r = lodo(cfg, extra, seeds)
        results[name] = {"cfg": cfg, "extra_train": list(extra), "seeds": list(seeds), **r}
        OUT.write_text(json.dumps(results, indent=1))
        print(f"{name:20s} direct macro-acc {100 * r['direct']['macro_acc']:5.1f}% (AUC {r['direct']['macro_auc']:.3f}, worst {100 * r['direct']['worst_acc']:.0f}%) | "
              f"all-conditions macro-acc {100 * r['all']['macro_acc']:5.1f}% (AUC {r['all']['macro_auc']:.3f}) | {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
