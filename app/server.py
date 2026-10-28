"""Web app: python app/server.py  ->  http://127.0.0.1:8008   (set PORT to change)"""
import os
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicedetect.detector import AudioError, Detector  # noqa: E402

app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"))
app.config["MAX_CONTENT_LENGTH"] = 26 * 1024 * 1024

_detector = None
_init_lock = threading.Lock()    # guards lazy model loading
_infer_lock = threading.Lock()   # serialises inference (separate lock: predict() calls get_detector() inside it)


def get_detector() -> Detector:
    global _detector
    with _init_lock:
        if _detector is None:
            _detector = Detector()
        return _detector


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    d = get_detector()
    return jsonify(ok=True, meta=d.meta)


@app.post("/api/predict")
def predict():
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="No audio received."), 400
    data = f.read()
    if not data:
        return jsonify(error="The audio file is empty."), 400
    suffix = Path(f.filename or "").suffix.lower()
    try:
        with _infer_lock:   # torch inference is not worth parallelising on a single-user local app
            result = get_detector().predict_bytes(data, suffix)
    except AudioError as e:
        return jsonify(error=str(e)), 422
    except Exception:  # noqa: BLE001
        app.logger.exception("prediction failed")
        return jsonify(error="Internal error while analysing the audio."), 500
    return jsonify(result)


SAMPLES = ROOT / "data" / "user_samples"
ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus"}


def _sample_counts() -> dict:
    return {lab: len([p for p in (SAMPLES / lab).glob("*") if p.is_file()]) if (SAMPLES / lab).exists() else 0
            for lab in ("real", "ai")}


@app.get("/api/samples")
def samples():
    return jsonify(_sample_counts())


@app.post("/api/sample")
def save_sample():
    """Store a labelled clip (stays on this computer) for tune_user.py to learn from."""
    label = request.form.get("label", "")
    f = request.files.get("audio")
    if label not in ("real", "ai") or f is None:
        return jsonify(error="Need an audio file and a label of 'real' or 'ai'."), 400
    data = f.read()
    ext = Path(f.filename or "").suffix.lower()
    ext = ext if ext in ALLOWED_EXT else ".bin"
    try:                                  # reject anything we could not train on later
        from voicedetect.audio import SR, load_bytes
        if len(load_bytes(data, ext)) < 1.5 * SR:
            return jsonify(error="That clip is shorter than 1.5 s - record or upload a longer one."), 422
    except Exception:  # noqa: BLE001
        return jsonify(error="Could not read this audio file."), 422
    d = SAMPLES / label
    d.mkdir(parents=True, exist_ok=True)
    import time
    name = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(3).hex()}{ext}"
    (d / name).write_bytes(data)
    return jsonify(saved=name, counts=_sample_counts())


@app.errorhandler(413)
def too_big(_):
    return jsonify(error="File is too large (limit 25 MB)."), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8008))
    get_detector()
    print(f"Open http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
