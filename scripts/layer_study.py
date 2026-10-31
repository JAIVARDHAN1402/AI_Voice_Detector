"""Does a shallower WavLM backbone (fewer transformer layers to download and run in the browser) cost accuracy?

Uses the cached 9-layer features, so no re-extraction. For each layer set the head is trained and evaluated exactly like
train.final (held-out speakers / recordings / generators). Nothing is saved; compare the printed numbers.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402

SETS = [[2, 3, 5, 7], [2, 3, 4], [2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]

for layers in (SETS if len(sys.argv) == 1 else [[int(x) for x in a.split(",")] for a in sys.argv[1:]]):
    _, rep, thr = train.final(layers, 0.05)
    p, pol = rep["pooled"], rep["policy"]
    print(f"LAYERS {layers}: backbone needs {max(layers)} transformer layers | pooled acc {p['acc']:.3f} AUC {p['auc']:.3f} EER {p['eer']:.3f} | "
          f"mlaad unseen {rep.get('mlaad_detection_unseen_generators', float('nan')):.3f} asv5 unseen {rep.get('asv5_fake_detection_unseen', float('nan')):.3f} "
          f"chatgpt unseen {rep.get('chatgpt_detection_unseen_voices', float('nan')):.3f} | real->real {pol['real_called_real_at_policy']:.3f} "
          f"AI->AI {pol['ai_called_ai_at_policy']:.3f}", flush=True)
