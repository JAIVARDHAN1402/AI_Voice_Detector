import time
from pathlib import Path
from huggingface_hub import snapshot_download
dst = Path(__file__).resolve().parents[1] / "data" / "raw" / "unidata"
for attempt in range(20):
    try:
        snapshot_download("UniDataPro/real-vs-fake-human-voice-deepfake-audio", repo_type="dataset", local_dir=dst)
        print("DONE"); break
    except Exception as e:
        print("retry", attempt, type(e).__name__, e); time.sleep(3)
