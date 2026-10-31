"""Web app: python app/server.py  ->  http://127.0.0.1:8008   (set PORT to change)

Use it from a phone on the same Wi-Fi:  HOST=0.0.0.0 python app/server.py  ->  https://<this computer's LAN IP>:8008
A phone browser only allows the microphone on HTTPS, so non-loopback mode serves a self-signed certificate (accept the browser
warning once). In that mode the "Teach it" endpoints that write files are disabled."""
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
HOST = os.environ.get("HOST", "127.0.0.1")
LOOPBACK = HOST in ("127.0.0.1", "localhost", "::1")


@app.before_request
def _no_writes_on_lan():
    if not LOOPBACK and request.path in ("/api/sample", "/api/samples"):
        return jsonify(error="Teaching is only available when the server runs on localhost."), 403
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


def lan_ips():
    import socket
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            if not info[4][0].startswith("127."):
                ips.add(info[4][0])
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:      # the address used for outbound traffic
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    return sorted(ips)


def self_signed(ips):
    """Self-signed certificate (valid for this computer's LAN addresses and localhost), kept in data/ (git-ignored)."""
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID
    d = ROOT / "data" / "lan_tls"
    d.mkdir(parents=True, exist_ok=True)
    cert_p, key_p, ips_p = d / "cert.pem", d / "key.pem", d / "ips.txt"
    if cert_p.exists() and key_p.exists() and ips_p.exists() and ips_p.read_text() == ",".join(ips):
        return str(cert_p), str(key_p)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AI Voice Detector (local)")])
    san = [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(i)) for i in ips]
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=365)).add_extension(x509.SubjectAlternativeName(san), critical=False)
            .sign(key, hashes.SHA256()))
    cert_p.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_p.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    ips_p.write_text(",".join(ips))
    return str(cert_p), str(key_p)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8008))
    get_detector()
    if LOOPBACK:
        print(f"Open http://127.0.0.1:{port}")
        app.run(host=HOST, port=port, threaded=True)
    else:
        ips = lan_ips()
        for ip in ips:
            print(f"Open on your phone (same Wi-Fi): https://{ip}:{port}   (accept the certificate warning once)")
        app.run(host=HOST, port=port, threaded=True, ssl_context=self_signed(ips))
