"""Layer 2 - speaker identity verification (ECAPA-TDNN, SpeechBrain) + voice-consistency analysis."""
import threading
from pathlib import Path

import numpy as np

SR = 16000


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


class SpeakerVerifier:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir) / "ecapa"
        self._enc = None
        self._lock = threading.Lock()

    def _encoder(self):
        with self._lock:
            if self._enc is None:
                from speechbrain.inference.speaker import EncoderClassifier
                src = str(self.dir) if (self.dir / "hyperparams.yaml").exists() else "speechbrain/spkrec-ecapa-voxceleb"
                self._enc = EncoderClassifier.from_hparams(source=src, savedir=str(self.dir), run_opts={"device": "cpu"})
        return self._enc

    def embed(self, wave: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            e = self._encoder().encode_batch(torch.from_numpy(np.asarray(wave, np.float32))[None, :])
        return _unit(e.squeeze().numpy().astype(np.float32))

    def embed_batch(self, waves) -> np.ndarray:
        import torch
        x = torch.from_numpy(np.stack([np.asarray(w, np.float32) for w in waves]))
        with torch.no_grad():
            e = self._encoder().encode_batch(x)
        return _unit(e.squeeze(1).numpy().astype(np.float32))

    @staticmethod
    def score(enrolled: np.ndarray, emb: np.ndarray) -> dict:
        """Cosine similarity to the centroid of the enrolled voiceprints (and to the closest single enrollment)."""
        c = _unit(enrolled.mean(axis=0))
        each = _unit(enrolled) @ emb
        return {"score": float(c @ emb), "best": float(each.max()), "n_enrolled": int(len(enrolled))}

    def consistency(self, wave: np.ndarray, emb_full: np.ndarray, seg_s: float = 2.0, hop_s: float = 1.0):
        """Does every 2 s segment sound like the same speaker as the whole response? Catches splices / morphing mid-sentence."""
        n, h = int(seg_s * SR), int(hop_s * SR)
        if len(wave) < n + h:
            return None
        starts = list(range(0, len(wave) - n + 1, h))
        segs = self.embed_batch([wave[s:s + n] for s in starts])
        sims = segs @ emb_full
        return {"min": float(sims.min()), "mean": float(sims.mean()), "segments": len(starts)}
