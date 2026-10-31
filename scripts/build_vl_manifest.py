"""Content-hash the in-browser VoiceLock model files and write web/vl/manifest.json.

  web/vl/ecapa.onnx                        -> ecapa.<hash>.onnx   (cached forever, a changed model can never be served stale)
  web/vl/whisper-tiny.en/onnx/*quantized*  -> sizes + a version hash (checked after download)
Run after scripts/export_ecapa_onnx.py (and after any change of the Whisper files).
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

VL = Path(__file__).resolve().parents[1] / "web" / "vl"


def sha(p):
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:8]


def main():
    plain = VL / "ecapa.onnx"
    if plain.exists():
        target = VL / f"ecapa.{sha(plain)}.onnx"
        os.replace(plain, target)
        for old in VL.glob("ecapa.*.onnx"):
            if old != target:
                old.unlink()
    found = sorted(VL.glob("ecapa.*.onnx"))
    if not found:
        raise SystemExit("no ecapa model: run scripts/export_ecapa_onnx.py first")
    wdir = VL / "whisper-tiny.en"
    shutil.rmtree(wdir / ".cache", ignore_errors=True)
    files = {f"onnx/{n}": (wdir / "onnx" / n).stat().st_size for n in ("encoder_model_quantized.onnx", "decoder_model_merged_quantized.onnx")}
    ver = hashlib.sha1(json.dumps(files, sort_keys=True).encode()).hexdigest()[:8]
    m = {"ecapa": {"file": found[0].name, "bytes": found[0].stat().st_size}, "whisper": files, "whisper_version": ver}
    (VL / "manifest.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
