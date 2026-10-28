"""Command line: python predict.py recording.wav [more files...]"""
import sys

from voicedetect.detector import AudioError, Detector


def main(paths):
    det = Detector()
    for p in paths:
        try:
            r = det.predict_file(p)
        except (AudioError, OSError) as e:
            print(f"{p}: ERROR - {e}")
            continue
        label = {"real": "REAL voice", "ai": "AI-generated", "unsure": "UNCERTAIN"}[r["verdict"]]
        print(f"{p}: {label}  (P(AI) = {r['prob_ai']:.2f}, {r['duration']:.1f}s, {r['n_windows']} segments)")
        for n in r["notes"]:
            print(f"   - {n}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
