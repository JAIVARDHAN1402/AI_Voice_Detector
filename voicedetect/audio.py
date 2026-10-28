"""Audio loading helpers: anything -> 16 kHz mono float32."""
import io
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000


def _resample(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=1) if y.ndim == 2 else y


def _ffmpeg_decode(path: str) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("Could not decode this audio format and ffmpeg is not installed.")
    cmd = [exe, "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def load_file(path) -> np.ndarray:
    """Load an audio file (wav/flac/ogg/mp3/m4a/webm...) as 16 kHz mono float32."""
    path = str(path)
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        return _resample(_to_mono(y), sr).astype(np.float32)
    except Exception:
        return _ffmpeg_decode(path)


def load_bytes(data: bytes, suffix: str = "") -> np.ndarray:
    """Load audio from raw bytes (e.g. an upload)."""
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return _resample(_to_mono(y), sr).astype(np.float32)
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / f"in{suffix or '.bin'}"
            p.write_bytes(data)
            return _ffmpeg_decode(str(p))


def windows(y: np.ndarray, win_s: float = 4.0, hop_s: float = 2.0, max_windows: int = 12):
    """Split a waveform into overlapping fixed windows (short clips are padded by tiling)."""
    n = int(win_s * SR)
    hop = int(hop_s * SR)
    if len(y) < n:
        reps = int(np.ceil(n / max(len(y), 1)))
        return [np.tile(y, reps)[:n]]
    starts = list(range(0, len(y) - n + 1, hop))
    if starts[-1] + n < len(y):
        starts.append(len(y) - n)
    if len(starts) > max_windows:
        starts = [starts[i] for i in np.linspace(0, len(starts) - 1, max_windows).astype(int)]
    return [y[s:s + n] for s in starts]
