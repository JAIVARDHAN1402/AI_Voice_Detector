"""Export the trained head for the browser: web/model/classifier.json (+ head.<hash>.bin for the MLP).

StandardScaler is folded into the first layer so the browser needs no separate normalisation step:
  linear head : logit = w . x + b,  w = coef / scale,  b = intercept - sum(w * mean)
  MLP head    : h = relu(x @ W1' + b1'), logit = h . w2 + b2,  W1' = W1 / scale[:, None], b1' = b1 - (mean / scale) @ W1
Re-run after every retrain (train.py final / tune_user.py), then redeploy the web app.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "model"
OUT = WEB / "classifier.json"


def hashed_model():
    """Give the ONNX file a content-hash name so it can be cached forever without ever serving a stale model."""
    plain = WEB / "wavlm_pooled.onnx"
    if plain.exists():
        h = hashlib.sha1(plain.read_bytes()).hexdigest()[:8]
        target = WEB / f"wavlm_pooled.{h}.onnx"
        os.replace(plain, target)
        for old in WEB.glob("wavlm_pooled.*.onnx"):
            if old != target:
                old.unlink()
    found = sorted(WEB.glob("wavlm_pooled.*.onnx"))
    if not found:
        sys.exit("no ONNX model in web/model - run export_onnx.py and quantize_onnx.py first")
    return found[0]


def main():
    bundle = sys.argv[1] if len(sys.argv) > 1 else "detector.joblib"       # e.g. detector_web.joblib for the shallower web backbone
    b = joblib.load(ROOT / "models" / bundle)
    clf = b["clf"]
    scaler, head = clf.steps[0][1], clf.steps[-1][1]
    mu, sd = scaler.mean_.astype(np.float64), scaler.scale_.astype(np.float64)
    rng = np.random.default_rng(0)
    X = rng.normal(mu, sd, size=(64, len(mu)))
    ref = clf.predict_proba(X)[:, 1]

    meta = {"unsure_margin": b.get("unsure_margin", 0.15)}
    for old in WEB.glob("head.*.bin"):
        old.unlink()
    if isinstance(head, MLPClassifier):
        W1, b1 = head.coefs_[0].astype(np.float64), head.intercepts_[0].astype(np.float64)
        W2, b2 = head.coefs_[1].astype(np.float64), head.intercepts_[1].astype(np.float64)
        W1f = W1 / sd[:, None]
        b1f = b1 - (mu / sd) @ W1
        z = np.maximum(X @ W1f + b1f, 0) @ W2 + b2
        err = np.abs(1 / (1 + np.exp(-z[:, 0])) - ref).max()
        print(f"folded MLP: {W1.shape[0]}x{W1.shape[1]} hidden, max |p error| vs scikit-learn = {err:.2e}")
        assert err < 1e-5, "folding mismatch"
        blob = np.concatenate([W1f.ravel(), b1f, W2.ravel(), b2]).astype("<f4").tobytes()
        hh = hashlib.sha1(blob).hexdigest()[:8]
        (WEB / f"head.{hh}.bin").write_bytes(blob)
        meta.update(type="mlp", hidden=int(W1.shape[1]), n_in=int(W1.shape[0]), head_file=f"head.{hh}.bin", head_bytes=len(blob))
    else:
        coef = head.coef_[0].astype(np.float64)
        w = coef / sd
        bias = float(head.intercept_[0] - np.sum(w * mu))
        err = np.abs(1 / (1 + np.exp(-(X @ w + bias))) - ref).max()
        print(f"folded linear head: {len(w)} weights, max |p error| vs scikit-learn = {err:.2e}")
        assert err < 1e-6, "folding mismatch"
        meta.update(type="linear", bias=bias, weights=[float(f"{x:.7g}") for x in w])

    model = hashed_model()
    real_max = b.get("real_max", 0.5 - meta["unsure_margin"])
    ai_min = b.get("ai_min", 0.5 + meta["unsure_margin"])
    OUT.write_text(json.dumps({
        "model_file": model.name, "model_bytes": model.stat().st_size,
        "layers": b["layers"], "dim": 1536, "threshold": b["threshold"], "real_max": real_max, "ai_min": ai_min,
        "window_s": 3.0, "hop_s": 1.5, "max_windows": 12, "meta_info": b.get("meta", {}), **meta,
    }))
    print("wrote", OUT, round(OUT.stat().st_size / 1e3), "KB", "| verdict band: real <=", round(real_max, 3), ", ai >=", round(ai_min, 3))


if __name__ == "__main__":
    main()
