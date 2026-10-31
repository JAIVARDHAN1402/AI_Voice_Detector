"""The confidence bands the detector reports come from reports/confidence_lodo.json (leave-one-corpus-out)."""
import json
from pathlib import Path

from voicedetect.detector import CONFIDENCE_FILE, load_confidence

ROOT = Path(__file__).resolve().parents[1]


def test_confidence_bands_are_consistent_and_loaded():
    c = load_confidence()
    assert c["real_z"] < 0 < c["ai_z"]
    assert 0.5 < c["low_acc"] < c["high_acc"] < 1.0           # confident verdicts must be measurably better than the rest
    assert 0.0 < c["high_acc"] <= 1.0


def test_missing_file_falls_back_to_the_default_bands(tmp_path):
    c = load_confidence(tmp_path / "nope.json")
    assert c == {"real_z": -5.0, "ai_z": 1.5}


def test_web_classifier_carries_the_same_bands():
    web = json.loads((ROOT / "web" / "model" / "classifier.json").read_text())["confidence"]
    ref = json.loads(CONFIDENCE_FILE.read_text())
    assert (web["real_z"], web["ai_z"]) == (ref["real_z"], ref["ai_z"])
    assert abs(web["high_acc"] - ref["direct"]["high_conf_acc"]) < 1e-3
