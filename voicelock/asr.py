"""Speech-to-text for checking that the spoken response contains the challenge (Whisper tiny.en, runs on CPU)."""
import threading
from pathlib import Path

import numpy as np


class Transcriber:
    def __init__(self, models_dir: Path):
        self.dir = Path(models_dir) / "whisper-tiny.en"
        self._pipe = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._pipe is None:
                from transformers import pipeline
                src = str(self.dir) if self.dir.exists() else "openai/whisper-tiny.en"
                self._pipe = pipeline("automatic-speech-recognition", model=src, device=-1)
        return self._pipe

    def transcribe(self, wave: np.ndarray) -> str:
        out = self._get()({"raw": np.asarray(wave, np.float32), "sampling_rate": 16000})
        return out["text"].strip()
