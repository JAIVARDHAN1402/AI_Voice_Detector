"""Convert the raw datasets into uniform 16 kHz mono FLAC clips + a metadata table.

Output: data/processed/<dataset>/<name>.flac and data/meta/<dataset>.csv with columns
    path, label (0 = real, 1 = AI/fake), dataset, group, system, split_hint
`group` identifies the speaker / source recording so train/test splits never share a
speaker or recording (otherwise a model can memorise voices instead of learning artefacts).
"""
import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicedetect.audio import SR, load_bytes, load_file  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
META_DIR = ROOT / "data" / "meta"   # one csv per dataset so parallel jobs never clobber each other
MAX_SECONDS = 12.0
FIELDS = ["path", "label", "dataset", "group", "system", "split_hint"]


def save_clip(y: np.ndarray, dataset: str, name: str) -> str:
    y = y[: int(MAX_SECONDS * SR)]
    d = OUT / dataset
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.flac"
    sf.write(p, y, SR, subtype="PCM_16", format="FLAC")
    return str(p.relative_to(ROOT)).replace("\\", "/")


class Meta:
    """Rows for the dataset currently being prepared; saved to data/meta/<dataset>.csv."""
    def __init__(self):
        self.rows = []
        self.ds = None

    def drop_dataset(self, ds):
        self.rows, self.ds = [], ds

    def add(self, **kw):
        self.rows.append({k: kw.get(k, "") for k in FIELDS})

    def save(self):
        META_DIR.mkdir(parents=True, exist_ok=True)
        with open(META_DIR / f"{self.ds}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)


def prep_garystafford(meta: Meta):
    meta.drop_dataset("gary")
    files = sorted((RAW / "garystafford").glob("train-*.parquet"))
    n = 0
    for fp in files:
        pf = pq.ParquetFile(fp)
        for batch in pf.iter_batches(batch_size=32, columns=["audio", "label"]):
            for a, lab in zip(batch.column("audio").to_pylist(), batch.column("label").to_pylist()):
                stem = Path(a["path"] or f"clip{n}").stem
                y = load_bytes(a["bytes"], ".flac")
                label = int(lab)  # dataset: 0 real, 1 fake
                if label == 1:
                    system = stem.split("_")[0]          # el / po / hg / hu / lv / sp (TTS vendor)
                    group = system
                else:
                    system = "real"
                    group = "yt_" + stem.split("_part")[0].split("_c")[0]   # source recording
                p = save_clip(y, "gary", stem)
                meta.add(path=p, label=label, dataset="gary", group=group, system=system)
                n += 1
        print(f"  {fp.name}: total clips so far {n}", flush=True)
    meta.save()


def prep_unidata(meta: Meta):
    meta.drop_dataset("unidata")
    root = RAW / "unidata"
    n = 0
    for spk in sorted(p for p in root.glob("*/*/*") if p.is_dir()):
        group = "uni_" + "_".join(spk.relative_to(root).parts)
        for f in sorted(spk.iterdir()):
            if f.suffix.lower() not in {".m4a", ".mp3", ".wav", ".flac", ".ogg"}:
                continue
            y = load_file(f)
            if len(y) < SR:
                continue
            real = f.stem.startswith("original")
            p = save_clip(y, "unidata", f"{'_'.join(spk.relative_to(root).parts)}_{f.stem}")
            meta.add(path=p, label=0 if real else 1, dataset="unidata", group=group,
                     system="real" if real else "synthetic")
            n += 1
    print(f"  unidata clips: {n}")
    meta.save()


def prep_asvspoof(meta: Meta, split: str, max_bonafide: int, max_spoof: int, tag: str):
    """Stream selected row groups of an ASVspoof2019-LA parquet shard from the Hub."""
    from huggingface_hub import HfFileSystem
    meta.drop_dataset(tag)
    fs = HfFileSystem()
    remote = f"datasets/Bisher/ASVspoof_2019_LA/data/{split}-00000-of-00001.parquet"
    nb = ns = 0
    with fs.open(remote, "rb", block_size=6 * 1024 * 1024) as fh:
        pf = pq.ParquetFile(fh)
        rg = pf.metadata.num_row_groups
        print(f"  {split}: {pf.metadata.num_rows} rows in {rg} row groups", flush=True)
        # Cheap pass over the label columns only, then fetch just the row groups we want:
        # bonafide groups spread across speakers, spoof groups stratified by attack system.
        rng = np.random.default_rng(0)
        comp = {}
        names = [pf.metadata.schema.column(j).name for j in range(pf.metadata.num_columns)]
        jk, js = names.index("key"), names.index("system_id")
        for i in range(rg):   # footer statistics only: no audio bytes are downloaded for this pass
            r = pf.metadata.row_group(i)
            ks, ss = r.column(jk).statistics, r.column(js).statistics
            comp[i] = ({ks.min, ks.max}, {ss.min, ss.max})
        bona = [i for i, (k, _) in comp.items() if k == {0}]
        spoof_by_sys = {}
        for i, (k, s) in comp.items():
            if k == {1} and len(s) == 1:
                spoof_by_sys.setdefault(next(iter(s)), []).append(i)
        n_bg = int(np.ceil(max_bonafide / 100))
        chosen = list(rng.choice(bona, min(n_bg, len(bona)), replace=False))
        per_sys = max(1, int(np.ceil(max_spoof / 100 / max(len(spoof_by_sys), 1))))
        for s, groups in sorted(spoof_by_sys.items()):
            chosen += list(rng.choice(groups, min(per_sys, len(groups)), replace=False))
        if not bona and not spoof_by_sys:      # shuffled split (e.g. ASVspoof test): every group is mixed
            n_groups = int(np.ceil(max_bonafide / 8))          # ~10% of clips are bonafide
            chosen = list(rng.choice(rg, min(n_groups, rg), replace=False))
        print(f"  fetching {len(chosen)} row groups ({len(bona)} bonafide groups, systems={sorted(spoof_by_sys)})", flush=True)
        for i in chosen:
            t = pf.read_row_group(int(i), columns=["speaker_id", "audio_file_name", "system_id", "key", "audio"])
            for sp, nm, sysid, key, a in zip(*(t.column(c).to_pylist() for c in
                                               ["speaker_id", "audio_file_name", "system_id", "key", "audio"])):
                label = int(key)  # 0 bonafide, 1 spoof
                if label == 0 and nb >= max_bonafide:
                    continue
                if label == 1 and ns >= max_spoof:
                    continue
                y = load_bytes(a["bytes"], ".flac")
                p = save_clip(y, tag, Path(nm).stem)
                meta.add(path=p, label=label, dataset=tag, group=f"vctk_{sp}" if label == 0 else sysid,
                         system="real" if label == 0 else sysid)
                nb += label == 0
                ns += label == 1
            print(f"  row group {i}: bonafide={nb} spoof={ns}", flush=True)
            meta.save()
    meta.save()


def prep_asv5(meta: Meta, shards=(3, 41, 77, 110, 150, 190), groups_per_shard=5):
    """ASVspoof 5 (2024) eval slice: modern TTS/VC attacks A17-A32 + 11 codec conditions + MLS audiobook bona fide."""
    import json
    import time
    from huggingface_hub import HfFileSystem
    meta.drop_dataset("asv5")
    fs = HfFileSystem()
    rng = np.random.default_rng(5)
    nb = ns = 0
    for sh in shards:
        remote = f"datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5/data/test-{sh:05d}-of-00200.parquet"
        for attempt in range(5):
            try:
                with fs.open(remote, "rb", block_size=6 * 1024 * 1024) as fh:
                    pf = pq.ParquetFile(fh)
                    for g in rng.choice(pf.metadata.num_row_groups, groups_per_shard, replace=False):
                        t = pf.read_row_group(int(g), columns=["label", "notes", "audio"])
                        for lab, note, a in zip(t.column("label").to_pylist(), t.column("notes").to_pylist(),
                                                t.column("audio").to_pylist()):
                            d = json.loads(note)
                            y = load_bytes(a["bytes"], ".flac")
                            name = d["utterance_id"]
                            p = save_clip(y, "asv5", name)
                            if lab == 0:
                                meta.add(path=p, label=0, dataset="asv5", group="mls_" + d["speaker_id"], system="real")
                                nb += 1
                            else:
                                meta.add(path=p, label=1, dataset="asv5", group=d["attack_id"], system=d["attack_id"])
                                ns += 1
                        print(f"  shard {sh} row group {g}: bonafide={nb} spoof={ns}", flush=True)
                        meta.save()
                break
            except Exception as e:  # noqa: BLE001  (network hiccups on long sessions)
                print(f"  shard {sh} attempt {attempt} failed: {type(e).__name__}; retrying", flush=True)
                time.sleep(3)
    meta.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gary", "unidata", "asv_train", "asv_val", "asv_test", "asv5"])
    ap.add_argument("--bonafide", type=int, default=1000)
    ap.add_argument("--spoof", type=int, default=1500)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = Meta()
    if a.what == "asv5":
        prep_asv5(m)
    elif a.what == "gary":
        prep_garystafford(m)
    elif a.what == "unidata":
        prep_unidata(m)
    else:
        split = {"asv_train": "train", "asv_val": "validation", "asv_test": "test"}[a.what]
        prep_asvspoof(m, split, a.bonafide, a.spoof, a.what)
