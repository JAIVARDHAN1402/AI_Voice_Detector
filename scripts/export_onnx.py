"""Export the feature extractor to ONNX for in-browser inference.

The wrapper takes one 3 s window (1 x 48000 float32, already peak-normalised) and returns the pooled features the
classifier uses: mean+std over time of the selected WavLM layers, shape (1, n_layers, 1536). Only the transformer
layers up to the deepest selected one are kept (layers 2,3,5,7 -> 7 layers), identical to what the Python pipeline computes.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import WavLMModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicedetect.features import MODEL_DIR  # noqa: E402

WIN = 48000


class Pooled(torch.nn.Module):
    def __init__(self, backbone, layers):
        super().__init__()
        self.m, self.layers = backbone, layers

    def forward(self, x):
        hs = self.m(x, output_hidden_states=True).hidden_states
        sel = torch.stack([hs[i] for i in self.layers], dim=1)          # (1, L, T, 768)
        return torch.cat([sel.mean(dim=2), sel.std(dim=2)], dim=-1)      # (1, L, 1536)


def build(layers):
    m = WavLMModel.from_pretrained(str(MODEL_DIR)).eval()
    keep = max(layers)
    m.encoder.layers = m.encoder.layers[:keep]
    m.config.num_hidden_layers = keep
    return Pooled(m, layers).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "web" / "model" / "wavlm_pooled_fp32.onnx"))
    ap.add_argument("--legacy", action="store_true", help="use the TorchScript exporter")
    a = ap.parse_args()
    layers = joblib.load(ROOT / "models" / "detector.joblib")["layers"]
    net = build(layers)
    x = torch.randn(1, WIN) * 0.1
    with torch.inference_mode():
        ref = net(x)
    print("torch output", tuple(ref.shape), "layers", layers)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(input_names=["audio"], output_names=["features"], opset_version=17)
    if a.legacy:
        torch.onnx.export(net, (x,), str(out), dynamo=False, **kwargs)
    else:
        torch.onnx.export(net, (x,), str(out), dynamo=True, **{**kwargs, "opset_version": 18})
    print("exported", out, round(out.stat().st_size / 1e6, 1), "MB")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"audio": x.numpy()})[0]
    d = np.abs(got - ref.numpy())
    print(f"parity vs torch: max abs diff {d.max():.2e}, mean abs diff {d.mean():.2e}")


if __name__ == "__main__":
    main()
