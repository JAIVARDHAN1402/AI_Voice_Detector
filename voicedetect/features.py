"""WavLM-based utterance embeddings.

Each 3 s window is encoded by a frozen pretrained WavLM backbone; for every transformer
layer we keep the mean and standard deviation over time. Different layers carry different
cues (low layers: channel/vocoder artefacts, high layers: linguistic/prosodic content), so
the classifier gets all of them and learns which are useful.
"""
from pathlib import Path

import numpy as np
import torch
from transformers import WavLMModel

from .audio import SR

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "wavlm-base-plus"
KEEP_LAYERS = 8        # only the lower transformer layers are used (cheaper, and where artefacts live)
N_LAYERS = KEEP_LAYERS + 1   # CNN-projection output + kept transformer layers
DIM = 768
FEAT_SHAPE = (N_LAYERS, 2 * DIM)


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    return y * (peak / m) if m > 1e-6 else y


class Embedder:
    def __init__(self, model_dir=MODEL_DIR, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.model = WavLMModel.from_pretrained(str(model_dir)).eval()
        self.model.encoder.layers = self.model.encoder.layers[:KEEP_LAYERS]
        self.model.config.num_hidden_layers = KEEP_LAYERS

    @torch.inference_mode()
    def embed(self, waves: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """waves: list of equal-length float32 arrays (16 kHz). Returns (N, 9, 1536) float32."""
        out = []
        for i in range(0, len(waves), batch_size):
            x = torch.from_numpy(np.stack([peak_normalize(w) for w in waves[i:i + batch_size]])).float()
            hs = self.model(x, output_hidden_states=True).hidden_states   # 9 x (B, T, 768)
            h = torch.stack(hs, dim=1)                                    # (B, 9, T, 768)
            feat = torch.cat([h.mean(dim=2), h.std(dim=2)], dim=-1)       # (B, 9, 1536)
            out.append(feat.numpy())
        return np.concatenate(out) if out else np.zeros((0, *FEAT_SHAPE), np.float32)
