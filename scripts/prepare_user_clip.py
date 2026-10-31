"""Turn one or more of YOUR OWN labelled files into the 'user' dataset (private: data/ is git-ignored).

  python scripts/prepare_user_clip.py --ai C:/path/clip.mp4 [--ai other.wav] [--real mine.wav]

Each file is trimmed and cut into overlapping 4 s segments (1 s apart). NOTE: overlapping segments of one file are not independent, so
there is no honest held-out test on the same file; check other voices for regressions instead.
Then: python scripts/extract_features.py user --max-win 4 --aug-frac 1.0 --fresh ; python scripts/extract_replay.py user
and retrain with scripts/train.py final (the 'user' set is up-weighted, see train.USER_WEIGHT).
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_cv import FIELDS, trim  # noqa: E402
from voicedetect.audio import SR, load_file  # noqa: E402

OUT = ROOT / "data" / "processed" / "user"
META = ROOT / "data" / "meta" / "user.csv"
SEG, HOP = 4 * SR, 1 * SR          # overlapping 4 s segments, 1 s apart (a single file is all we have; more samples of the same voice)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", action="append", default=[], help="file(s) that are AI voices")
    ap.add_argument("--real", action="append", default=[], help="file(s) that are real human voices")
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows = []
    for label, files in ((1, a.ai), (0, a.real)):
        for f in files:
            y = trim(load_file(f))
            gid = f"user_{'ai' if label else 'real'}_{Path(f).stem}"
            for s in range(0, max(1, len(y) - SEG + 1), HOP):
                seg = y[s:s + SEG]
                if len(seg) < 2 * SR:
                    continue
                name = f"{len(rows):04d}"
                sf.write(OUT / f"{name}.flac", seg.astype(np.float32), SR, subtype="PCM_16", format="FLAC")
                rows.append(dict(path=f"data/processed/user/{name}.flac", label=label, dataset="user",
                                 group=gid, system="user_ai" if label else "user_real"))
    META.parent.mkdir(parents=True, exist_ok=True)
    with open(META, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)
    print(f"{len(rows)} segments: {sum(r['label'] == 1 for r in rows)} AI, {sum(r['label'] == 0 for r in rows)} real")


if __name__ == "__main__":
    main()
