"""Build the 'cv' dataset: real volunteer recordings (Common Voice 17, English) vs. modern neural TTS of the same sentences.

Real  : fixie-ai/common_voice_17_0 (en/test) - thousands of speakers on their own laptop/phone microphones.
Fake  : Microsoft neural voices through the `edge-tts` package (47 English voices, many accents), reading the
        *same sentences* so that wording can't be used to tell the classes apart. Both sides are MP3 and
        both are silence-trimmed, so neither codec nor leading silence is a shortcut.
"""
import asyncio
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicedetect.audio import SR, load_bytes  # noqa: E402

OUT = ROOT / "data" / "processed" / "cv"
META = ROOT / "data" / "meta" / "cv.csv"
FIELDS = ["path", "label", "dataset", "group", "system", "split_hint"]
SHARDS = [0, 5, 10, 15]
N_CLIPS = 420


def trim(y: np.ndarray) -> np.ndarray:
    t, _ = librosa.effects.trim(y, top_db=30)
    return np.pad(t, (int(0.1 * SR), int(0.1 * SR)))


def fetch_real(rng):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    rows = []
    for s in SHARDS:
        remote = f"datasets/fixie-ai/common_voice_17_0/en/test-{s:05d}-of-00019.parquet"
        with fs.open(remote, "rb", block_size=4 * 1024 * 1024) as fh:
            pf = pq.ParquetFile(fh)
            for g in rng.choice(pf.metadata.num_row_groups, 2, replace=False):
                t = pf.read_row_group(int(g), columns=["client_id", "audio", "sentence"])
                for cid, a, sent in zip(t.column("client_id").to_pylist(), t.column("audio").to_pylist(),
                                        t.column("sentence").to_pylist()):
                    rows.append((cid, a["bytes"], sent))
                print(f"  shard {s} row group {g}: {len(rows)} clips so far", flush=True)
    return rows


async def synth_all(jobs, cache_dir, concurrency=3):
    """Synthesise (text, voice) jobs; mp3 bytes are cached on disk so an interrupted run resumes."""
    import edge_tts
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def stream_one(text, voice):
        buf = bytearray()
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                buf += chunk["data"]
        return bytes(buf)

    async def one(i, text, voice):
        nonlocal done
        f = cache_dir / f"{i:04d}.mp3"
        if f.exists() and f.stat().st_size > 0:
            return
        async with sem:
            for attempt in range(4):
                try:
                    data = await asyncio.wait_for(stream_one(text, voice), timeout=30)   # a hung socket must not block the run
                    if data:
                        f.write_bytes(data)
                        break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(1 + attempt)
            else:
                print(f"  TTS failed for job {i} ({voice})", flush=True)
        done += 1
        if done % 20 == 0:
            print(f"  synthesised {done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(i, t, v) for i, (t, v) in enumerate(jobs)))


def main():
    rng = np.random.default_rng(0)
    OUT.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    real_json = ROOT / "data" / "cv_real.json"
    if real_json.exists():
        saved = json.loads(real_json.read_text())
        meta, kept_sentences = saved["meta"], saved["sentences"]
        print(f"reusing {len(meta)} real clips from {real_json.name}", flush=True)
    else:
        print("Fetching Common Voice clips ...", flush=True)
        rows = fetch_real(rng)
        rng.shuffle(rows)
        meta, kept_sentences = [], []
        per_speaker = {}
        for cid, b, sent in rows:
            if len(meta) >= N_CLIPS:
                break
            if per_speaker.get(cid, 0) >= 3 or len(sent) < 25:         # many speakers, no repeated short phrases
                continue
            try:
                y = trim(load_bytes(b, ".mp3"))
            except Exception:  # noqa: BLE001
                continue
            if not (2.0 <= len(y) / SR <= 12.0):
                continue
            per_speaker[cid] = per_speaker.get(cid, 0) + 1
            name = f"real_{len(meta):04d}"
            sf.write(OUT / f"{name}.flac", y, SR, subtype="PCM_16", format="FLAC")
            meta.append(dict(path=f"data/processed/cv/{name}.flac", label=0, dataset="cv",
                             group="cv_" + cid[:12], system="real"))
            kept_sentences.append(sent)
        real_json.write_text(json.dumps({"meta": meta, "sentences": kept_sentences}))
        print(f"real clips: {len(meta)} from {len(per_speaker)} speakers", flush=True)

    import edge_tts
    voices = [v["ShortName"] for v in asyncio.run(edge_tts.list_voices()) if v["Locale"].startswith("en-")]
    jobs = [(s, voices[int(rng.integers(len(voices)))]) for s in kept_sentences]
    print(f"synthesising {len(jobs)} sentences with {len(voices)} voices ...", flush=True)
    cache = ROOT / "data" / "cv_tts_cache"
    cache.mkdir(exist_ok=True)
    asyncio.run(synth_all(jobs, cache))
    nf = 0
    for i, (text, voice) in enumerate(jobs):
        f = cache / f"{i:04d}.mp3"
        if not f.exists():
            continue
        try:
            y = trim(load_bytes(f.read_bytes(), ".mp3"))
        except Exception:  # noqa: BLE001
            continue
        if len(y) / SR < 1.5:
            continue
        name = f"tts_{i:04d}"
        sf.write(OUT / f"{name}.flac", y, SR, subtype="PCM_16", format="FLAC")
        meta.append(dict(path=f"data/processed/cv/{name}.flac", label=1, dataset="cv",
                         group=voice, system="edge:" + voice.split("-")[0] + "-" + voice.split("-")[1]))
        nf += 1
    print(f"fake clips: {nf}", flush=True)
    with open(META, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in meta:
            w.writerow({k: m.get(k, "") for k in FIELDS})
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
