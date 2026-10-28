"""Fold StandardScaler + LogisticRegression into one linear layer and write web/model/classifier.json.

logit = sum_i coef_i * (x_i - mean_i) / scale_i + intercept = w . x + b,   w_i = coef_i / scale_i,  b = intercept - sum_i w_i * mean_i
Re-run this after every retrain (tune_user.py) and redeploy the web app to ship the new classifier.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web" / "model" / "classifier.json"


def hashed_model():
    """Give the ONNX file a content-hash name so it can be cached forever without ever serving a stale model."""
    d = ROOT / "web" / "model"
    plain = d / "wavlm_pooled.onnx"
    if plain.exists():
        h = hashlib.sha1(plain.read_bytes()).hexdigest()[:8]
        target = d / f"wavlm_pooled.{h}.onnx"
        os.replace(plain, target)
        for old in d.glob("wavlm_pooled.*.onnx"):
            if old != target:
                old.unlink()
    found = sorted(d.glob("wavlm_pooled.*.onnx"))
    if not found:
        sys.exit("no ONNX model in web/model - run export_onnx.py and quantize_onnx.py first")
    return found[0]


def main():
    b = joblib.load(ROOT / "models" / "detector.joblib")
    scaler, lr = b["clf"].named_steps["standardscaler"], b["clf"].named_steps["logisticregression"]
    coef = lr.coef_[0].astype(np.float64)
    w = coef / scaler.scale_
    bias = float(lr.intercept_[0] - np.sum(w * scaler.mean_))

    rng = np.random.default_rng(0)
    X = rng.normal(scaler.mean_, scaler.scale_, size=(64, len(w)))
    err = np.abs(X @ w + bias - b["clf"].decision_function(X)).max()
    print(f"folded classifier: {len(w)} weights, max |logit error| vs scikit-learn = {err:.2e}")
    assert err < 1e-6, "folding mismatch"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = hashed_model()
    OUT.write_text(json.dumps({
        "model_file": model.name, "model_bytes": model.stat().st_size,
        "layers": b["layers"], "dim": 1536, "threshold": b["threshold"], "unsure_margin": b.get("unsure_margin", 0.15),
        "window_s": 3.0, "hop_s": 1.5, "max_windows": 12, "bias": bias,
        "weights": [float(f"{x:.7g}") for x in w], "meta": b.get("meta", {}),
    }))
    print("wrote", OUT, round(OUT.stat().st_size / 1e3), "KB")


if __name__ == "__main__":
    sys.exit(main())
