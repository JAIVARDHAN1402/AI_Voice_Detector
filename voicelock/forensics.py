"""Layer 1 - AI-voice forensics.

authenticity : the project's WavLM + MLP real-vs-synthetic detector (trained on ASVspoof 2019/5, MLAAD, OpenAI voices, ... and
               on simulated speaker->microphone replay audio, so it also sees many replay and voice-conversion artefacts).
acoustic     : cheap signal-level checks. These are heuristics, NOT a trained replay detector: they only tighten the
               authenticity limit a little and are shown to the operator as context.
"""
import threading

import numpy as np

SR = 16000


def acoustic_analysis(y: np.ndarray) -> dict:
    y = np.asarray(y, np.float32)
    n = int(0.025 * SR)
    frames = y[: len(y) // n * n].reshape(-1, n)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    floor_db, top_db = float(np.percentile(db, 10)), float(np.percentile(db, 95))
    silence = float((rms < 1e-5).mean())
    clip = float((np.abs(y) >= 0.999).mean())
    spec = np.abs(np.fft.rfft(frames * np.hanning(n), axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / SR)
    tot = spec.sum() + 1e-12
    high = float(spec[:, freqs >= 4000].sum() / tot)
    cum = np.cumsum(spec.sum(axis=0)) / tot
    rolloff = float(freqs[min(np.searchsorted(cum, 0.95), len(freqs) - 1)])
    dyn = top_db - floor_db

    flags, score = [], 0.0
    if silence > 0.05:
        flags.append("digital silence (exact zeros) - typical of synthesised or spliced audio"); score += 0.35
    if floor_db < -80:
        flags.append("unnaturally clean noise floor"); score += 0.25
    if rolloff < 3500 and high < 0.002:
        flags.append("strongly band-limited (phone / loudspeaker / re-recording)"); score += 0.15
    if floor_db > -40 and dyn < 20:
        flags.append("very noisy, low dynamic range (possible loudspeaker re-recording, heuristic)"); score += 0.20
    if clip > 0.01:
        flags.append("clipping"); score += 0.10
    return {"anomaly_score": round(min(score, 1.0), 2), "flags": flags, "noise_floor_db": round(floor_db, 1),
            "dynamic_range_db": round(dyn, 1), "rolloff_hz": round(rolloff), "highband_ratio": round(high, 4),
            "digital_silence": round(silence, 3), "clipping": round(clip, 4)}


class Forensics:
    def __init__(self):
        self._det = None
        self._lock = threading.Lock()

    def _detector(self):
        with self._lock:
            if self._det is None:
                from voicedetect.detector import Detector
                self._det = Detector()
        return self._det

    def analyze(self, wave: np.ndarray) -> dict:
        r = self._detector().predict_wave(wave)
        return {"p_ai": float(r["prob_ai"]), "verdict": r["verdict"],
                "windows": [{"start": w["start"], "p_ai": w["prob_ai"]} for w in r["windows"] if not w["skipped"]],
                "acoustic": acoustic_analysis(wave)}
