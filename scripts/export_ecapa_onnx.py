"""Export the SpeechBrain ECAPA-TDNN speaker encoder (fbank + sentence mean-norm + embedding model) to ONNX for the in-browser VoiceLock.

Input  audio: (1, N) float32, 16 kHz mono, any length >= ~0.5 s.   Output: (1, 192) raw (not yet length-normalised) embedding,
identical to `EncoderClassifier.encode_batch`. Large weights are stored as float16 on disk (a Cast restores float32), like the WavLM export.
Compares with SpeechBrain on real clips (cosine similarity of the two embeddings must be ~1).
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from voicelock.config import Settings  # noqa: E402
from voicelock.speaker import SpeakerVerifier  # noqa: E402


class Wrap(torch.nn.Module):
    """SpeechBrain's Fbank with torch.stft replaced by an explicit DFT (conv1d with a windowed cos/sin basis): same numbers, but exportable."""

    def __init__(self, enc, n_fft=400, hop=160):
        super().__init__()
        fb = enc.mods.compute_features
        self.fbank, self.n, self.e = fb.compute_fbanks, enc.mods.mean_var_norm, enc.mods.embedding_model
        self.n_fft, self.hop = n_fft, hop
        win = torch.hamming_window(n_fft)                                   # periodic, like SpeechBrain's STFT default
        k = torch.arange(n_fft // 2 + 1).float()[:, None]
        t = torch.arange(n_fft).float()[None, :]
        ang = 2 * np.pi * k * t / n_fft
        self.register_buffer("basis", torch.cat([win * torch.cos(ang), -win * torch.sin(ang)], dim=0)[:, None, :])   # (2*(n_fft/2+1), 1, n_fft)

    def features(self, wav):
        x = torch.nn.functional.pad(wav, (self.n_fft // 2, self.n_fft // 2))   # center=True, zero padding
        y = torch.nn.functional.conv1d(x[:, None, :], self.basis, stride=self.hop)
        h = self.n_fft // 2 + 1
        spec = (y[:, :h] ** 2 + y[:, h:] ** 2).transpose(1, 2)                  # (B, T, 201) power spectrogram
        return self.fbank(spec)

    def forward(self, wav):
        lens = torch.ones(wav.shape[0])
        return self.e(self.n(self.features(wav), lens), lens).squeeze(1)


def main():
    out = ROOT / "data" / "vl_build" / "ecapa_fp32.onnx"
    final = ROOT / "web" / "vl" / "ecapa.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    sv = SpeakerVerifier(Settings().models_dir)
    enc = sv._encoder()
    enc.mods.eval()
    net = Wrap(enc).eval()
    x = torch.randn(1, 16000 * 4) * 0.1
    with torch.inference_mode():
        d = (net.features(x) - enc.mods.compute_features(x)).abs()
        print(f"fbank vs SpeechBrain: max abs diff {d.max():.2e}, mean {d.mean():.2e} (features span ~80 dB)")
        ref = net(x)
    print("torch output", tuple(ref.shape))
    torch.onnx.export(net, (x,), str(out), input_names=["audio"], output_names=["embedding"], opset_version=17, dynamo=False,
                      dynamic_axes={"audio": {1: "n"}})
    print("exported", round(out.stat().st_size / 1e6, 1), "MB")
    sys.path.insert(0, str(ROOT / "scripts"))
    from quantize_onnx import store_fp16
    store_fp16(str(out), str(final))
    print("fp16-stored:", round(final.stat().st_size / 1e6, 1), "MB ->", final)

    import onnxruntime as ort
    from voicedetect.audio import load_file
    sess = ort.InferenceSession(str(final), providers=["CPUExecutionProvider"])
    import pandas as pd
    m = pd.read_csv(ROOT / "data" / "meta" / "libri.csv")
    paths = m[m.label == 0].path.tolist()[:12]
    cos = []
    for p in paths:
        y = load_file(ROOT / p)[: 16000 * 6].astype(np.float32)
        a = sess.run(None, {"audio": y[None]})[0].reshape(-1)
        b = enc.encode_batch(torch.from_numpy(y)[None]).squeeze().detach().numpy()
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    print("cosine(onnx, speechbrain) over", len(cos), "real clips: min %.5f mean %.5f" % (min(cos), np.mean(cos)))
    for L in (1.0, 2.5, 9.0):                                  # different lengths work (dynamic axis)
        y = load_file(ROOT / paths[0])[: int(16000 * L)].astype(np.float32)
        print(f"  length {L}s ->", sess.run(None, {"audio": y[None]})[0].shape)


if __name__ == "__main__":
    main()
