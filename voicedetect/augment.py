"""Channel augmentations that mimic real-world capture: codecs (what a browser mic/WebM-Opus,
phone calls and MP3 uploads do to audio), background noise, room reverb, band-limiting.

Applied to *both* classes so the model cannot use "was compressed" as a shortcut for "real"/"fake".
"""
import shutil
import subprocess

import numpy as np

from .audio import SR

FFMPEG = shutil.which("ffmpeg")


def codec_roundtrip(y: np.ndarray, codec: str, bitrate_k: int) -> np.ndarray:
    """Encode + decode through ffmpeg. codec in {opus, mp3, aac}."""
    if FFMPEG is None:
        return y
    spec = {
        "opus": ["-c:a", "libopus", "-f", "ogg"],
        "mp3": ["-c:a", "libmp3lame", "-f", "mp3"],
        "aac": ["-c:a", "aac", "-f", "adts"],
    }[codec]
    raw = y.astype(np.float32).tobytes()
    enc = subprocess.run([FFMPEG, "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                          *spec, "-b:a", f"{bitrate_k}k", "-"], input=raw, capture_output=True)
    if enc.returncode != 0 or not enc.stdout:
        return y
    dec = subprocess.run([FFMPEG, "-v", "error", "-i", "-", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         input=enc.stdout, capture_output=True)
    out = np.frombuffer(dec.stdout, dtype=np.float32).copy()
    if len(out) < 0.5 * len(y):
        return y
    # codecs add delay/padding; trim or pad to the original length
    return out[: len(y)] if len(out) >= len(y) else np.pad(out, (0, len(y) - len(out)))


def add_noise(y, rng):
    snr = rng.uniform(8, 35)
    kind = rng.choice(["white", "pink", "hum"])
    n = rng.standard_normal(len(y)).astype(np.float32)
    if kind == "pink":
        f = np.fft.rfft(n)
        f /= np.sqrt(np.maximum(np.arange(len(f)), 1))
        n = np.fft.irfft(f, len(y)).astype(np.float32)
    elif kind == "hum":
        t = np.arange(len(y)) / SR
        n = (np.sin(2 * np.pi * rng.choice([50, 60]) * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
             + 0.3 * n).astype(np.float32)
    p_s = np.mean(y ** 2) + 1e-9
    p_n = np.mean(n ** 2) + 1e-9
    return y + n * np.sqrt(p_s / (p_n * 10 ** (snr / 10)))


def reverb(y, rng):
    rt60 = rng.uniform(0.15, 0.6)
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir = (ir / np.linalg.norm(ir)).astype(np.float32)
    wet = np.convolve(y, ir)[: len(y)]
    mix = rng.uniform(0.3, 0.8)
    return (1 - mix) * y + mix * wet.astype(np.float32)


def band_limit(y, rng):
    import librosa
    low = int(rng.choice([4000, 6000, 8000]))   # simulate narrow-band capture
    d = librosa.resample(y, orig_sr=SR, target_sr=low * 2, res_type="soxr_hq") if low < 8000 else y
    return librosa.resample(d, orig_sr=low * 2, target_sr=SR, res_type="soxr_hq")[: len(y)] if low < 8000 else y


def augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = y.astype(np.float32)
    ops = []
    if rng.random() < 0.5:
        ops.append("noise")
    if rng.random() < 0.3:
        ops.append("reverb")
    if rng.random() < 0.2:
        ops.append("band")
    for op in ops:
        y = {"noise": add_noise, "reverb": reverb, "band": band_limit}[op](y, rng)
    if rng.random() < 0.75:
        codec = rng.choice(["opus", "opus", "mp3", "aac"])
        br = int(rng.choice({"opus": [16, 24, 32, 48, 64], "mp3": [32, 48, 64, 96, 128],
                             "aac": [24, 32, 48, 64, 96]}[codec]))
        y = codec_roundtrip(y, codec, br)
    y = y * float(rng.uniform(0.2, 1.0))
    return np.clip(y, -1, 1).astype(np.float32)


def replay(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate audio played from a (small) speaker in a room and re-recorded by a microphone:
    speaker band-limit -> room reverb -> mic noise -> browser/phone codec -> level change.
    Applied to both classes in training so 'sounds re-recorded' can't mean 'real'."""
    from scipy.signal import butter, sosfilt
    s = float(rng.uniform(0, 1))                                   # 0 = mild, 1 = harsh
    lo, hi = float(rng.uniform(120, 350)), float(rng.uniform(3500, 7500))
    y = sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), y).astype(np.float32)
    n = int(rng.uniform(0.2, 0.6) * SR)
    t = np.arange(n) / SR
    ir = (rng.standard_normal(n) * np.exp(-6.9 * t / (n / SR))).astype(np.float32)
    ir[0] = 1.0
    ir /= np.linalg.norm(ir)
    mix = 0.35 + 0.35 * s
    y = (1 - mix) * y + mix * np.convolve(y, ir)[: len(y)]
    snr = float(rng.uniform(8, 30))
    z = rng.standard_normal(len(y)).astype(np.float32)
    y = y + z * np.sqrt((np.mean(y ** 2) + 1e-9) / ((np.mean(z ** 2) + 1e-9) * 10 ** (snr / 10)))
    if rng.random() < 0.85:
        y = codec_roundtrip(y, str(rng.choice(["opus", "opus", "aac", "mp3"])), int(rng.choice([24, 32, 48, 64])))
    if rng.random() < 0.3:                                         # crude AGC / limiter
        y = np.tanh(y * float(rng.uniform(1.5, 4.0)) / (np.abs(y).max() + 1e-6))
    y = y * float(rng.uniform(0.2, 1.0))
    return np.clip(y, -1, 1).astype(np.float32)
