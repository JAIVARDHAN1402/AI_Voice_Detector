"""int8-quantise the transformer matmuls of the exported model, then compare predictions with the fp32 PyTorch model.

    python scripts/quantize_onnx.py
The comparison runs the real Detector pipeline's windows through (a) PyTorch fp32 features and (b) the quantised ONNX
model, pushes both through the classifier, and reports how much the per-window logit and the final verdicts move.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicedetect.audio import load_file, windows  # noqa: E402
from voicedetect.features import Embedder, peak_normalize  # noqa: E402

TMP = ROOT / "data" / "onnx"            # intermediates (not deployed)
WEB = ROOT / "web" / "model"            # what gets deployed


def store_fp16(src, dst, min_elems=100_000):
    """Keep large float32 weights (the convolutions) as float16 on disk; a Cast node restores float32 at run time."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    m = onnx.load(src)
    casts = []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= min_elems:
            arr = numpy_helper.to_array(init)
            half = numpy_helper.from_array(arr.astype(np.float16), init.name + "_fp16")
            casts.append((init, half))
    for init, half in casts:
        m.graph.initializer.remove(init)
        m.graph.initializer.append(half)
        m.graph.node.insert(len(casts) and 0, helper.make_node("Cast", [half.name], [init.name], to=TensorProto.FLOAT,
                                                              name="cast_" + init.name))
    onnx.checker.check_model(m)
    onnx.save(m, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(TMP / "wavlm_pooled_fp32.onnx"))
    ap.add_argument("--dst", default=str(TMP / "wavlm_pooled_int8.onnx"))
    ap.add_argument("--web", default=str(WEB / "wavlm_pooled.onnx"), help="final file: int8 matmuls + fp16-stored conv weights")
    ap.add_argument("--skip-quantize", action="store_true")
    ap.add_argument("--bundle", default="detector.joblib", help="model bundle under models/ used for the accuracy comparison")
    a = ap.parse_args()
    if not a.skip_quantize:
        quantize_dynamic(a.src, a.dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                         use_external_data_format=False)
    print("quantised model:", round(Path(a.dst).stat().st_size / 1e6, 1), "MB")
    Path(a.web).parent.mkdir(parents=True, exist_ok=True)
    store_fp16(a.dst, a.web)
    print("web model (fp16-stored conv weights):", round(Path(a.web).stat().st_size / 1e6, 1), "MB")
    a.dst = a.web

    bundle = joblib.load(ROOT / "models" / a.bundle)
    layers, clf = bundle["layers"], bundle["clf"]
    emb = Embedder()
    sess = ort.InferenceSession(a.dst, providers=["CPUExecutionProvider"])

    files = (sorted((ROOT / "data/ood/real").glob("*.flac"))[:24] + sorted((ROOT / "data/ood/sapi").glob("*.wav"))[:8]
             + sorted((ROOT / "data/ood/gtts").glob("*.mp3"))[:8])
    d_logit, flips, wins = [], 0, 0
    for f in files:
        for w in windows(load_file(f), 3.0, 3.0, max_windows=2):
            w = peak_normalize(w).astype(np.float32)
            ref = emb.embed([w])[0][layers].reshape(1, -1)                       # fp32 PyTorch features
            got = sess.run(None, {"audio": w[None]})[0].reshape(1, -1)           # int8 ONNX features
            lr = clf.decision_function(ref)[0]
            lg = clf.decision_function(got)[0]
            d_logit.append(lg - lr)
            flips += (lr > 0) != (lg > 0)
            wins += 1
    d = np.array(d_logit)
    print(f"{wins} windows: logit shift mean {d.mean():+.3f}, std {d.std():.3f}, max |shift| {np.abs(d).max():.3f}; "
          f"decision flips {flips} ({flips / wins:.1%})")


if __name__ == "__main__":
    main()
