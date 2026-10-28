"""Inference: audio -> verdict. Shared by the web app and the command-line tool."""
from pathlib import Path

import joblib
import numpy as np

from .audio import SR, load_bytes, load_file, windows
from .features import Embedder, ROOT

BUNDLE = ROOT / "models" / "detector.joblib"
WIN_S, HOP_S = 3.0, 1.5
MAX_SECONDS = 60.0
MIN_SECONDS = 1.0


class AudioError(ValueError):
    """User-facing problem with the submitted audio."""


def _speech_fraction(w: np.ndarray) -> float:
    """Fraction of 25 ms frames that are well above the clip's noise floor (crude energy VAD)."""
    n = int(0.025 * SR)
    f = w[: len(w) // n * n].reshape(-1, n)
    e = np.sqrt((f ** 2).mean(axis=1) + 1e-12)
    floor = max(np.percentile(e, 10), 1e-4)
    return float((e > max(3.0 * floor, 0.004)).mean())


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Detector:
    def __init__(self, bundle_path: Path = BUNDLE):
        b = joblib.load(bundle_path)
        self.clf = b["clf"]
        self.layers = b["layers"]
        self.threshold = b["threshold"]       # decision boundary on P(AI) that balances the two error types
        self.unsure_margin = b.get("unsure_margin", 0.15)
        self.meta = b.get("meta", {})
        self.embedder = Embedder()

    def _prob_windows(self, ws):
        feats = self.embedder.embed(ws)[:, self.layers, :].reshape(len(ws), -1)
        return self.clf.predict_proba(feats)[:, 1]

    def predict_wave(self, y: np.ndarray) -> dict:
        dur = len(y) / SR
        if dur < MIN_SECONDS:
            raise AudioError("The audio is too short. Please provide at least 1-2 seconds of speech.")
        if dur > MAX_SECONDS:
            y = y[: int(MAX_SECONDS * SR)]
        if not np.isfinite(y).all():
            raise AudioError("The audio contains invalid samples.")
        # Trim leading/trailing silence (training clips were trimmed the same way) but never below 1 s.
        import librosa
        t, _ = librosa.effects.trim(y, top_db=35)
        if len(t) >= MIN_SECONDS * SR:
            pad = int(0.1 * SR)
            y = np.pad(t, (pad, pad))
            dur = len(y) / SR
        ws = windows(y, WIN_S, HOP_S, max_windows=12)
        starts = []
        if len(y) >= int(WIN_S * SR):
            n = int(WIN_S * SR)
            starts = list(range(0, len(y) - n + 1, int(HOP_S * SR)))
            if starts[-1] + n < len(y):
                starts.append(len(y) - n)
            if len(starts) > 12:
                starts = [starts[i] for i in np.linspace(0, len(starts) - 1, 12).astype(int)]
        else:
            starts = [0]
        starts = [s / SR for s in starts]

        speech = [_speech_fraction(w) >= 0.15 and float(np.abs(w).max()) > 0.003 for w in ws]
        if not any(speech):
            raise AudioError("No speech detected - the recording seems silent or too quiet.")
        idx = [i for i, ok in enumerate(speech) if ok]
        probs = self._prob_windows([ws[i] for i in idx])

        pw = [{"start": float(starts[i]), "skipped": True, "prob_ai": 0.0} for i in range(len(ws))]
        for i, p in zip(idx, probs):
            pw[i].update(skipped=False, prob_ai=float(p))

        # Aggregate in logit space (robust to one extreme window); shift so the calibrated threshold maps to 0.5.
        z = float(np.mean(_logit(probs)) - _logit(self.threshold))
        p_ai = float(1 / (1 + np.exp(-z)))
        lo, hi = 0.5 - self.unsure_margin, 0.5 + self.unsure_margin
        verdict = "ai" if p_ai >= hi else "real" if p_ai <= lo else "unsure"
        for w in pw:
            if not w["skipped"]:
                w["prob_ai"] = float(1 / (1 + np.exp(-(_logit(w["prob_ai"]) - _logit(self.threshold)))))

        notes = []
        speech_dur = len(idx) * HOP_S if len(ws) > 1 else dur
        if dur < 5:
            notes.append("Short clip: accuracy improves a lot with 6-10+ seconds of continuous speech.")
        if len(idx) < len(ws):
            notes.append(f"{len(ws) - len(idx)} silent segment(s) were ignored.")
        live = np.array([w["prob_ai"] for w in pw if not w["skipped"]])
        if len(live) > 1 and (live.max() >= hi) and (live.min() <= lo):
            notes.append("Segments disagree: part of the clip looks real and part looks AI. It may be spliced or mixed.")
        if verdict == "unsure":
            notes.append("The model is not confident. Try a longer, cleaner recording.")
        notes.append("Heavily compressed audio (e.g. voice messages, phone calls) is harder to judge reliably.")
        return {"verdict": verdict, "prob_ai": p_ai, "duration": dur, "n_windows": len(ws),
                "windows": pw, "notes": notes}

    def predict_bytes(self, data: bytes, suffix: str = "") -> dict:
        try:
            y = load_bytes(data, suffix)
        except Exception as e:  # noqa: BLE001
            raise AudioError("Could not read this audio file. Try WAV, MP3, M4A, FLAC or OGG.") from e
        return self.predict_wave(y)

    def predict_file(self, path) -> dict:
        return self.predict_wave(load_file(path))
