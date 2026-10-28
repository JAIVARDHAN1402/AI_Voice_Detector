"""Tune the detector on your own recordings.

1. Record / upload clips in the app's "Teach it" tab (saved to data/user_samples/{real,ai}/).
2. python scripts/tune_user.py

It (a) converts your clips, (b) scores them with the CURRENT model so you can see how it does on your mic,
(c) retrains with your clips included (weighted up), and (d) reports before/after on your clips that were
held out of training. The previous model is kept as models/detector_base.joblib; use --dry-run to only report.
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402
from voicedetect.audio import SR, load_file  # noqa: E402
from voicedetect.augment import codec_roundtrip  # noqa: E402

SAMPLES = ROOT / "data" / "user_samples"
OUT = ROOT / "data" / "processed" / "user"
META = ROOT / "data" / "meta" / "user.csv"
MODELS = ROOT / "models"
FIELDS = ["path", "label", "dataset", "group", "system", "split_hint"]


def convert(samples: Path):
    """Raw clips -> trimmed 16 kHz FLAC. Non-browser clips get an Opus round-trip so 'came from the mic' (WebM/Opus)
    can't be a shortcut for 'real'."""
    import librosa
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    for label, lab in (("real", 0), ("ai", 1)):
        for i, f in enumerate(sorted((samples / label).glob("*"))):
            try:
                y = load_file(f)
            except Exception as e:  # noqa: BLE001
                print(f"  skipping {f.name}: {e}")
                continue
            t, _ = librosa.effects.trim(y, top_db=35)
            y = np.pad(t if len(t) > SR else y, (int(0.1 * SR), int(0.1 * SR)))[: 20 * SR]
            if len(y) < 1.5 * SR:
                continue
            if f.suffix.lower() not in {".webm", ".opus"}:
                y = codec_roundtrip(y, "opus", int(rng.choice([24, 32, 48, 64])))
            name = f"{label}_{i:04d}"
            sf.write(OUT / f"{name}.flac", y, SR, subtype="PCM_16", format="FLAC")
            rows.append(dict(path=f"data/processed/user/{name}.flac", label=lab, dataset="user",
                             group=f"user_{name}", system="real" if lab == 0 else "user_ai"))
    with open(META, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-weight", type=float, default=3.0, help="extra weight on your clips")
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not replace the model")
    ap.add_argument("--samples-dir", default=str(SAMPLES), help="folder with real/ and ai/ subfolders")
    a = ap.parse_args()

    rows = convert(Path(a.samples_dir))
    n_real = sum(r["label"] == 0 for r in rows)
    n_ai = len(rows) - n_real
    print(f"{n_real} real-voice clips, {n_ai} AI-voice clips")
    if min(n_real, n_ai) < 6:
        sys.exit("Need at least 6 clips of each kind (20+ is much better). Record more in the 'Teach it' tab.")

    print("embedding your clips (a few minutes) ...", flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "extract_features.py"), "user", "--threads", "12",
                    "--max-win", "4", "--aug-frac", "1.0", "--fresh"], check=True)

    base = joblib.load(MODELS / "detector.joblib")
    layers = base["layers"]
    data = {n: train.load(n) for n in train.available(train.TRAIN_SETS + train.EXTRA_SETS)}
    user = train.load("user")
    data["user"] = user

    # hold out ~30% of your clips (whole clips, never windows of a clip used in training)
    train_part, test_part = train.split_holdout(user, "user", frac=0.3, seed=1)
    parts = {n: train.split_holdout(d, n)[0] for n, d in data.items() if n != "user"}
    parts["user"] = train_part

    def report(name, clf, thr):
        out = {}
        for aug, tag in ((False, "clean"), (True, "augmented")):
            m = test_part["aug"] == aug
            p = train.predict(clf, train.take(test_part, m), layers)
            y = test_part["y"][m]
            pred = (p >= thr).astype(int)
            out[tag] = (float((pred[y == 0] == 0).mean()) if (y == 0).any() else float("nan"),
                        float((pred[y == 1] == 1).mean()) if (y == 1).any() else float("nan"), int(len(y)))
        print(f"  {name:22s} " + "   ".join(
            f"{t}: real voice kept {r:.0%} / AI caught {f:.0%} (n={n})" for t, (r, f, n) in out.items()))

    print("\nOn YOUR held-out clips (window level):")
    report("current model", base["clf"], base["threshold"])

    X = train.concat(list(parts.values()))
    w = train.cell_weights(X["ds"], X["y"])
    w = w * np.where(X["ds"] == "user", a.user_weight, 1.0)
    clf = train.make_clf(a.C)
    clf.fit(train.feats(X["X"], layers), X["y"], logisticregression__sample_weight=w)
    # reuse the old threshold only as a starting point: recompute it on pooled held-out data incl. yours
    held = train.concat([train.split_holdout(d, n)[1] for n, d in data.items() if n != "user"] + [test_part])
    p_held = train.predict(clf, held, layers)
    _, thr = train.eer(held["y"], p_held)
    report("tuned model", clf, thr)
    print(f"  pooled held-out accuracy of tuned model at its threshold: "
          f"{((p_held >= thr).astype(int) == held['y']).mean():.1%} (threshold {thr:.2f})")

    if a.dry_run:
        print("\n--dry-run: model not replaced.")
        return
    # final refit on everything, including the held-out parts
    Xall = train.concat(list(data.values()))
    wall = train.cell_weights(Xall["ds"], Xall["y"]) * np.where(Xall["ds"] == "user", a.user_weight, 1.0)
    final = train.make_clf(a.C)
    final.fit(train.feats(Xall["X"], layers), Xall["y"], logisticregression__sample_weight=wall)
    shutil.copy(MODELS / "detector.joblib", MODELS / "detector_base.joblib")
    meta = dict(base.get("meta", {}))
    meta["tuned_on_user_clips"] = {"real": n_real, "ai": n_ai}
    joblib.dump({"clf": final, "layers": layers, "threshold": float(thr), "unsure_margin": base.get("unsure_margin", 0.15),
                 "meta": meta}, MODELS / "detector.joblib")
    print("\nSaved models/detector.joblib (previous model kept as models/detector_base.joblib). Restart the app to use it.")


if __name__ == "__main__":
    main()
