"""Build the 'chatgpt' dataset: fake speech from OpenAI / ChatGPT voices.

Sources (all fake; real speech comes from the other datasets):
  - ata990/AdvBench-Audio-GPT4o-TTS (eng): GPT-4o TTS voices alloy, echo, fable, nova, onyx, shimmer
  - LALM-emotional-damage/gpt-4o-mini-tts (MIT): expressive gpt-4o-mini-tts clips
  - leafspark/openai-voices (Apache-2.0): recordings of the ChatGPT app voices Juniper / Sky (clean reads only;
    the screen recordings are skipped because they may also contain the user's own voice)
`system` = the voice / source, so evaluation can hold whole voices out.
"""
import csv
import io
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_cv import FIELDS, trim  # noqa: E402
from voicedetect.audio import SR, load_bytes, load_file  # noqa: E402

OUT = ROOT / "data" / "processed" / "chatgpt"
RAW = ROOT / "data" / "raw" / "chatgpt"
META = ROOT / "data" / "meta" / "chatgpt.csv"
VOICES = ["alloy", "echo", "fable", "nova", "onyx", "shimmer"]
PER_VOICE = 110
PER_EMOTION_SET = 240
LONG_FILES = ["juniper/cppstory.wav", "juniper/industrialRevolution.wav", "sky/statement.wav",
              "sky/clonedgum.SYNTH.wav", "sky/drama_0.SYNTH.wav", "sky/speechify_sky.SYNTH.wav", "sky/speechify_sky2.SYNTH.wav"]

meta = []


def keep(y, group, system):
    if not (1.5 <= len(y) / SR <= 14):
        return
    name = f"{len(meta):05d}"
    sf.write(OUT / f"{name}.flac", y[: 12 * SR], SR, subtype="PCM_16", format="FLAC")
    sid = re.sub(r"[^A-Za-z0-9._-]+", "_", system)
    meta.append(dict(path=f"data/processed/chatgpt/{name}.flac", label=1, dataset="chatgpt", group=re.sub(r"[^A-Za-z0-9._-]+", "_", group), system=sid))


def from_parquet(repo_path, column, system, n, rng, group=None):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    pf = pq.ParquetFile(HfFileSystem().open(f"datasets/{repo_path}", "rb"))
    rows = []
    for g in range(pf.metadata.num_row_groups):
        rows.append(pf.metadata.row_group(g).num_rows)
    total = sum(rows)
    pick = set(rng.choice(total, min(n, total), replace=False).tolist())
    base, done = 0, 0
    for g, cnt in enumerate(rows):
        want = sorted(i - base for i in pick if base <= i < base + cnt)
        base += cnt
        if not want:
            continue
        tab = pf.read_row_group(g, columns=[column])
        col = tab.column(column).to_pylist()
        for i in want:
            cell = col[i]
            try:
                y = trim(load_bytes(cell["bytes"], ".wav"))
            except Exception:  # noqa: BLE001
                continue
            keep(y, group or system, system)
            done += 1
    print(f"  {system}: {done} clips", flush=True)


def from_long(rel, rng):
    from huggingface_hub import hf_hub_download
    local = hf_hub_download("leafspark/openai-voices", rel, repo_type="dataset", local_dir=RAW)
    y = load_file(local)
    system = "chatgpt_app_" + rel.split("/")[0] + ("_synth" if "SYNTH" in rel else "")
    group = rel.replace("/", "_")
    step = 8 * SR
    n = 0
    for s in range(0, max(1, len(y) - step // 2), step):
        seg = trim(y[s:s + step])
        if len(seg) >= 3 * SR:
            keep(seg, group, system)
            n += 1
    print(f"  {rel}: {n} segments", flush=True)


def main():
    rng = np.random.default_rng(23)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    print("GPT-4o TTS voices", flush=True)
    for v in VOICES:
        from_parquet(f"ata990/AdvBench-Audio-GPT4o-TTS/eng/{v}-00000-of-00001.parquet", "audio_prompt", f"gpt4o_tts_{v}", PER_VOICE, rng)
    print("gpt-4o-mini-tts (expressive)", flush=True)
    from_parquet("LALM-emotional-damage/gpt-4o-mini-tts/data/train-00000-of-00001.parquet", "audio", "gpt4o_mini_tts", PER_EMOTION_SET, rng)
    print("ChatGPT app voices", flush=True)
    for rel in LONG_FILES:
        from_long(rel, rng)
    META.parent.mkdir(parents=True, exist_ok=True)
    with open(META, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: m.get(k, "") for k in FIELDS} for m in meta)
    print(f"DONE {len(meta)} clips, systems: {sorted(set(m['system'] for m in meta))}", flush=True)


if __name__ == "__main__":
    main()
