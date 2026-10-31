"""python -m voicelock  ->  http://127.0.0.1:8010"""
import os
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voicelock.api import create_app  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8010)))
