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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TRAIN_SETS = ["gary", "unidata", "asv_train", "cv", "libri", "asv5", "mlaad", "chatgpt", "user"]   # "user": your own labelled clips (prepare_user_clip.py); skipped when absent
HEAD = "mlp"                       # "mlp" (one 256-unit hidden layer) or "linear" (logistic regression)
# newest TTS systems in MLAAD held out entirely from the evaluation fit (honest "never seen this generator" test)
HOLD_TTS = {"Cartesia.ai_Sonic-3_", "Higgs-Audio-V2", "Microsoft_VibeVoice_Large", "Index-TTS-2.0", "DeepGram", "sesame_csm",
            "Chatterbox", "minimax_speech-2.6-hd", "Qwen2.5-Omni", "Maya1_TTS", "Kyutai-TTS", "SoulX-Podcast"}
HOLD_GPT = {"gpt4o_tts_onyx", "gpt4o_tts_shimmer", "chatgpt_app_juniper"}   # OpenAI / ChatGPT voices held out of the evaluation fit
USER_WEIGHT = 8.0                  # your own clips count this many times more than their share (there are very few of them)
FAKE_ONLY = {"mlaad": HOLD_TTS, "chatgpt": HOLD_GPT, "user": set()}   # fake-only sets: hold out whole generators + 20% of clips of the others
HOLD_ATTACKS = {"A29", "A30", "A31", "A32"}   # ASVspoof 5 systems held out entirely in the evaluation fit
EXTRA_SETS = ["asv_test"]          # unseen speakers + unseen attack systems (A07-A19)
USE_REPLAY = True                  # include speaker->room->mic degraded samples when their features exist


def load(name):
    """-> dict with one entry per *sample*: X (clean and augmented copies stacked), y, group, ds, aug flag."""
    z = np.load(FEAT / f"{name}.npz")
    ai = z["aug_idx"]
    y, grp = z["y"], np.array([re.sub(r"_p2$", "", g) for g in z["group"]])
    d = {
        "X": np.concatenate([z["Xc"], z["Xa"]]),
        "y": np.concatenate([y, y[ai]]),
        "group": np.concatenate([grp, grp[ai]]),
        "system": np.concatenate([z["system"], z["system"][ai]]),
        "clip": np.concatenate([z["clip"], z["clip"][ai]]),
        "aug": np.concatenate([np.zeros(len(y), bool), np.ones(len(ai), bool)]),
        "replay": np.zeros(len(y) + len(ai), bool),
    }
    rp = FEAT / f"{name}_replay.npz"           # speaker -> room -> microphone copies (extract_replay.py)
    if rp.exists() and USE_REPLAY:
        r = np.load(rp)
        n = len(r["y"])
        d["X"] = np.concatenate([d["X"], r["Xr"]])
        d["y"] = np.concatenate([d["y"], r["y"]])
        d["group"] = np.concatenate([d["group"], np.array([re.sub(r"_p2$", "", g) for g in r["group"]])])
        d["system"] = np.concatenate([d["system"], r["system"]])
        d["clip"] = np.concatenate([d["clip"], r["clip"]])
        d["aug"] = np.concatenate([d["aug"], np.ones(n, bool)])
        d["replay"] = np.concatenate([d["replay"], np.ones(n, bool)])
    d["ds"] = np.array([name] * len(d["y"]))
    return d


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
    if HEAD == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=0.05, early_stopping=True, n_iter_no_change=6,
                                                             max_iter=80, random_state=0))
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


def fit_weighted(clf, X, y, w):
    clf.fit(X, y, **{f"{clf.steps[-1][0]}__sample_weight": w})
    return clf


def fit(d, layers, C):
    w = cell_weights(d["ds"], d["y"]) * np.where(d["ds"] == "user", USER_WEIGHT, 1.0)
    return fit_weighted(make_clf(C), feats(d["X"], layers), d["y"], w)


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
    parts = {n: split_holdout(data[n], n) for n in available(TRAIN_SETS) if n not in FAKE_ONLY}
    for n, hold_sys in FAKE_ONLY.items():
        if n in data:
            d = data[n]
            rng = np.random.default_rng(3)
            clips = np.unique(d["clip"])
            test_c = set(rng.choice(clips, int(len(clips) * 0.2), replace=False))
            te = np.isin(d["clip"], list(test_c)) | np.isin(d["system"], list(hold_sys))
            parts[n] = (take(d, ~te), take(d, te))
    if "asv5" in parts:                        # attack systems A29-A32 are never trained on in the evaluation fit
        tr, te = parts["asv5"]
        hold = np.isin(tr["system"], list(HOLD_ATTACKS))
        parts["asv5"] = (take(tr, ~hold), concat([te, take(tr, hold)]))
    train = concat([p[0] for p in parts.values()])
    clf = fit(train, layers, C)

    report = {"layers": layers, "C": C, "train_samples": int(len(train["y"])), "eval": {}}
    pooled_p, pooled_y = [], []
    evals = {f"{n} (held-out speakers/recordings)": p[1] for n, p in parts.items()}
    for n in available(EXTRA_SETS):
        evals[f"{n} (unseen attacks + speakers)"] = data[n]
    cats = (("clean", lambda d: ~d["aug"] & ~d["replay"]), ("augmented", lambda d: d["aug"] & ~d["replay"]),
            ("replay (speaker->room->mic)", lambda d: d["replay"]))
    cat_preds = {}
    for name, d in evals.items():
        for cname, sel in cats:
            m = sel(d)
            if not m.any():
                continue
            p = predict(clf, take(d, m), layers)
            cat_preds[(name, cname)] = (p, d["y"][m])
            pooled_p.append(p); pooled_y.append(d["y"][m])
    pp, yy = np.concatenate(pooled_p), np.concatenate(pooled_y)
    _, thr = eer(yy, pp)                       # one threshold for everything, incl. re-recorded audio (what a mic gives)
    for (name, cname), (p, y) in cat_preds.items():
        r = metrics(y, p, thr)
        if r:
            report["eval"][f"{name} / {cname}"] = r
    report["pooled"] = metrics(yy, pp, thr) | {"threshold": thr}
    # per-attack-system detection rate on unseen ASVspoof attacks
    if "asv_test" in data:
        d = data["asv_test"]
        p = predict(clf, d, layers)
        report["asv_test_per_system_fake_detection"] = {
            s: float((p[(d["system"] == s)] >= thr).mean()) for s in sorted(set(d["system"])) if s != "real"}

    if "asv5" in parts:                        # modern attacks: held-out clips of seen systems vs. never-seen systems A29-A32
        te = parts["asv5"][1]
        fake = te["y"] == 1
        pr = predict(clf, te, layers)
        report["asv5_per_attack_fake_detection"] = {
            s + (" (never trained on)" if s in HOLD_ATTACKS else ""): float((pr[fake & (te["system"] == s)] >= thr).mean())
            for s in sorted(set(te["system"][fake]))}
        for lab, sel in (("seen systems", ~np.isin(te["system"], list(HOLD_ATTACKS))), ("unseen systems A29-A32", np.isin(te["system"], list(HOLD_ATTACKS)))):
            m = fake & sel
            report["asv5_fake_detection_" + lab.split()[0]] = float((pr[m] >= thr).mean())

    if "mlaad" in parts:                       # recent TTS systems: detection per generator on held-out clips / generators
        te = parts["mlaad"][1]
        pr = predict(clf, te, layers)
        det = pr >= thr
        rows = {s_ + (" (never trained on)" if s_ in HOLD_TTS else ""): float(det[te["system"] == s_].mean())
                for s_ in sorted(set(te["system"]))}
        report["mlaad_per_system_detection"] = dict(sorted(rows.items(), key=lambda kv: kv[1]))
        unseen = np.isin(te["system"], list(HOLD_TTS))
        report["mlaad_detection_seen_generators"] = float(det[~unseen].mean())
        report["mlaad_detection_unseen_generators"] = float(det[unseen].mean())

    if "chatgpt" in parts:                     # OpenAI / ChatGPT voices: per voice, held-out clips and never-trained voices
        te = parts["chatgpt"][1]
        det = predict(clf, te, layers) >= thr
        report["chatgpt_per_voice_detection"] = {
            s_ + (" (never trained on)" if s_ in HOLD_GPT else ""): float(det[te["system"] == s_].mean()) for s_ in sorted(set(te["system"]))}
        unseen = np.isin(te["system"], list(HOLD_GPT))
        report["chatgpt_detection_seen_voices"] = float(det[~unseen].mean())
        report["chatgpt_detection_unseen_voices"] = float(det[unseen].mean())

    # Verdict policy from held-out scores (calibrated P(AI) = sigmoid(logit(p) - logit(thr))): call "real" only if few
    # held-out AI windows score that low, and "ai" only if few held-out real windows score that high.
    lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))  # noqa: E731
    calib = 1 / (1 + np.exp(-(lg(pp) - lg(thr))))
    real_max = float(np.clip(np.quantile(calib[yy == 1], 0.08), 0.03, 0.45))
    ai_min = 0.5   # chosen by hand: the calibrated threshold itself (the quantile policy gave ~0.7 and left real ChatGPT audio "uncertain")
    report["policy"] = {"real_max": real_max, "ai_min": ai_min,
                        "ai_called_real_at_policy": float((calib[yy == 1] <= real_max).mean()),
                        "real_called_ai_at_policy": float((calib[yy == 0] >= ai_min).mean()),
                        "real_called_real_at_policy": float((calib[yy == 0] <= real_max).mean()),
                        "ai_called_ai_at_policy": float((calib[yy == 1] >= ai_min).mean())}

    # refit on every training sample (including the held-out parts and the unseen-attack set) for deployment
    full = concat([data[n] for n in data])
    clf_full = fit(full, layers, C)
    return clf_full, report, thr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["explore", "final"])
    ap.add_argument("--layers", type=int, nargs="+", default=[2, 3, 5, 7])
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--out", default="detector.joblib", help="bundle file name under models/ (the web build uses detector_web.joblib)")
    ap.add_argument("--report", default="evaluation.json", help="report file name under reports/")
    a = ap.parse_args()
    if a.mode == "explore":
        explore()
    else:
        clf, report, thr = final(a.layers, a.C)
        REPORTS.mkdir(exist_ok=True)
        MODELS.mkdir(exist_ok=True)
        (REPORTS / a.report).write_text(json.dumps(report, indent=2))
        joblib.dump({"clf": clf, "layers": a.layers, "threshold": float(thr), "unsure_margin": 0.15,
                     "real_max": report["policy"]["real_max"], "ai_min": report["policy"]["ai_min"],
                     "meta": {"head": HEAD, "backbone": "microsoft/wavlm-base-plus (8 layers)", "datasets": TRAIN_SETS + EXTRA_SETS,
                              "pooled_eer": report["pooled"]["eer"]}}, MODELS / a.out)
        print(json.dumps(report, indent=2))
