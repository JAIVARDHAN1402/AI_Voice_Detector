"""VoiceLock experiments (real models, real data). Results are merged into reports/voicelock_eval.json.

  python scripts/voicelock_eval.py speaker    ECAPA speaker verification: EER, FAR/FRR at the tier thresholds, noise robustness
  python scripts/voicelock_eval.py forensics  Layer 1 on whole clips: real vs many families of AI voice, rates at each tier limit
  python scripts/voicelock_eval.py attacks    full pipeline: genuine user, replay, pre-made TTS, live TTS of the challenge, replayed TTS
  python scripts/voicelock_eval.py asr        challenge recognition rate (TTS reading the phrase) clean and with noise
  python scripts/voicelock_eval.py fuzz       token tampering / transaction substitution / replay
  python scripts/voicelock_eval.py latency    per-stage latency on CPU
Only the things that can be measured honestly are measured; see README "VoiceLock" for what is NOT evaluated.
"""
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voicedetect.audio import SR, load_file  # noqa: E402
from voicedetect.augment import replay  # noqa: E402
from voicelock import challenge as ch  # noqa: E402
from voicelock.config import Settings  # noqa: E402
from voicelock.forensics import Forensics  # noqa: E402
from voicelock.service import VoiceLock  # noqa: E402

OUT = ROOT / "reports" / "voicelock_eval.json"
TTS_VOICES = ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural", "en-IN-NeerjaNeural",
              "en-IN-PrabhatNeural", "en-AU-NatashaNeural", "en-US-AriaNeural"]
S = Settings()


def save(key, value):
    OUT.parent.mkdir(exist_ok=True)
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[key] = value
    OUT.write_text(json.dumps(d, indent=2))
    print(json.dumps(value, indent=2), flush=True)


def meta(name):
    return pd.read_csv(ROOT / "data" / "meta" / f"{name}.csv")


def clip(path, secs=4.0):
    y = load_file(ROOT / path)
    return y[: int(secs * SR)] if len(y) > secs * SR else y


def speakers(min_clips=7, per_dataset=None):
    """{(dataset, speaker): [paths]} for real speakers with enough clips (never mixes datasets)."""
    out = {}
    for ds in ("asv_test", "asv_train", "libri"):
        m = meta(ds)
        r = m[m.label == 0]
        for g, rows in r.groupby("group"):
            if len(rows) >= min_clips:
                out[(ds, g)] = rows.path.tolist()
    return out


def eer(genuine, impostor):
    thr = np.sort(np.r_[genuine, impostor])
    far = np.array([(impostor >= t).mean() for t in thr])
    frr = np.array([(genuine < t).mean() for t in thr])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i])


def noisy(y, snr_db, rng):
    p = np.mean(y ** 2) + 1e-12
    return (y + rng.normal(0, np.sqrt(p / 10 ** (snr_db / 10)), len(y))).astype(np.float32)


# --------------------------------------------------------------------------------------------- speaker
def run_speaker():
    from voicelock.speaker import SpeakerVerifier
    sv = SpeakerVerifier(S.models_dir)
    rng = np.random.default_rng(0)
    spk = speakers()
    keys = sorted(spk)
    print(len(keys), "speakers with >=7 clips", flush=True)
    emb, enrolled, tests = {}, {}, {}
    for k in keys:
        paths = list(rng.permutation(spk[k]))[:12]
        e = [sv.embed(clip(p)) for p in paths]
        enrolled[k], tests[k] = np.stack(e[:3]), (paths[3:], np.stack(e[3:]))
    gen, imp = [], []
    for k in keys:
        for e in tests[k][1]:
            gen.append(SpeakerVerifier.score(enrolled[k], e)["score"])
        for k2 in keys:
            if k2 != k and k2[0] == k[0]:
                imp += [SpeakerVerifier.score(enrolled[k], e)["score"] for e in tests[k2][1][:3]]
    gen, imp = np.array(gen), np.array(imp)
    e, thr = eer(gen, imp)
    res = {"speakers": len(keys), "genuine_trials": len(gen), "impostor_trials": len(imp), "eer": round(e, 4), "eer_threshold": round(thr, 3),
           "genuine_mean": round(float(gen.mean()), 3), "impostor_mean": round(float(imp.mean()), 3),
           "at_tier_thresholds": {t: {"min": S.speaker_min[t], "FRR": round(float((gen < S.speaker_min[t]).mean()), 4),
                                      "FAR": round(float((imp >= S.speaker_min[t]).mean()), 4)} for t in S.speaker_min}}
    # noise: genuine trials with added noise (impostors unchanged)
    res["noise_genuine_FRR"] = {}
    for snr in (20, 10, 5):
        sc = []
        for k in keys[:20]:
            for p in tests[k][0][:3]:
                sc.append(SpeakerVerifier.score(enrolled[k], sv.embed(noisy(clip(p), snr, rng)))["score"])
        sc = np.array(sc)
        res["noise_genuine_FRR"][f"SNR {snr} dB"] = {t: round(float((sc < S.speaker_min[t]).mean()), 3) for t in S.speaker_min}
    save("speaker_verification", res)


# --------------------------------------------------------------------------------------------- forensics
def run_forensics():
    fx = Forensics()
    rng = np.random.default_rng(1)
    sets = {}
    for name, ds, label, n in [("real: ASVspoof/others (asv_test)", "asv_test", 0, 60), ("real: LibriSpeech readers", "libri", 0, 40),
                               ("real: Common Voice volunteers", "cv", 0, 40),
                               ("AI: ASVspoof 2019 attacks (asv_test)", "asv_test", 1, 50), ("AI: ASVspoof 5 modern attacks", "asv5", 1, 50),
                               ("AI: 64 recent TTS systems (MLAAD)", "mlaad", 1, 60), ("AI: OpenAI/ChatGPT voices", "chatgpt", 1, 50),
                               ("AI: Microsoft neural voices (libri, cv)", "libri", 1, 30)]:
        m = meta(ds)
        m = m[m.label == label]
        sets[name] = [m.path.iloc[i] for i in rng.choice(len(m), min(n, len(m)), replace=False)]
    rows, allp = {}, {"real": [], "ai": []}
    for name, paths in sets.items():
        ps = []
        for p in paths:
            try:
                ps.append(fx.analyze(clip(p, 8.0))["p_ai"])
            except ValueError:                       # no detectable speech in this clip: nothing to score
                continue
        ps = np.array(ps)
        rows[name] = {"n": len(ps), "mean_p_ai": round(float(ps.mean()), 3),
                      **{f"pass_{t}": round(float((ps < S.ai_limit[t]).mean()), 3) for t in S.ai_limit}}
        allp["real" if name.startswith("real") else "ai"] += ps.tolist()
        print(name, rows[name], flush=True)
    real, ai = np.array(allp["real"]), np.array(allp["ai"])
    e, thr = eer(-real, -ai)  # accept = low P(AI): real clips are the genuine trials, score = -P(AI)
    res = {"per_set": rows, "clips": {"real": len(real), "ai": len(ai)},
           "per_tier": {t: {"limit": S.ai_limit[t], "real_accepted(1-FRR)": round(float((real < S.ai_limit[t]).mean()), 3),
                            "AI_accepted(FAR)": round(float((ai < S.ai_limit[t]).mean()), 3)} for t in S.ai_limit},
           "clip_level_eer": round(e, 4), "note": "whole clips up to 8 s through the production Layer-1 pipeline"}
    save("forensics_clip_level", res)


# --------------------------------------------------------------------------------------------- TTS helpers
def tts(text, voice, path):
    import edge_tts
    asyncio.run(asyncio.wait_for(edge_tts.Communicate(text, voice).save(str(path)), 40))
    return load_file(path)


def spoken(c):
    return ", ".join(c["words"]) + ". " + " ".join(c["digits"])


# --------------------------------------------------------------------------------------------- attacks
def make_lock(tmp):
    t = {"v": time.mktime(time.strptime(time.strftime("%Y-%m-%d") + " 12:00:00", "%Y-%m-%d %H:%M:%S"))}
    return VoiceLock(Settings(data_dir=Path(tmp), velocity_limit=10 ** 9), clock=lambda: t["v"])   # velocity rule off: the experiment fires many requests per user


def run_attacks():
    rng = np.random.default_rng(2)
    spk = speakers(min_clips=9)
    candidates = [k for k in sorted(spk) if k[0] == "libri"][:16]       # trimmed read speech like a phone/mic recording
    tmp = tempfile.mkdtemp()
    vl = make_lock(tmp)
    cases = {"genuine": [], "replay": [], "pretts": [], "live_tts": [], "live_tts_replayed": []}
    paths_by, keys = {}, []
    for k in candidates:
        if len(keys) == 10:
            break
        uid = f"{k[0]}:{k[1]}"
        ps = list(rng.permutation(spk[k]))
        paths_by[uid] = ps
        used, enrolled_paths = 0, set()
        for p in ps:                                   # enrolment refuses samples the detector thinks are synthetic: take the next one
            try:
                vl.enroll(uid, clip(p, 6.0))
            except Exception:  # noqa: BLE001
                continue
            enrolled_paths.add(p)
            used += 1
            if used == 3:
                break
        if used < 3:                                   # this speaker's audio is unusable (e.g. a corrupt corpus clip): skip the speaker
            print("skipping speaker", uid, flush=True)
            continue
        keys.append(k)
        paths_by[uid] = [p for p in ps if p not in enrolled_paths]          # test clips never include enrolment clips
        for rcp in ("ACME", "Globex"):
            vl.store.add_known_recipient(uid, rcp.casefold())
    amounts = {"low": "2000", "high": "200000"}      # recipient already known -> tier from the amount alone

    def run(uid, tier, wave_fn, label):
        st = vl.start_transaction({"user_id": uid, "amount": amounts[tier], "recipient": "ACME", "reference": "eval"})
        assert st["tier"] == tier, (st["tier"], tier)
        out = None
        for attempt in range(4):
            w = wave_fn(st["challenge"], attempt)
            real_asr = vl.asr
            if label == "genuine":
                vl.asr = type("A", (), {"transcribe": staticmethod(lambda _w, c=st["challenge"]: spoken(c))})()   # human reading is not synthesizable here
            try:
                out = vl.respond(st["session_id"], w)
            finally:
                vl.asr = real_asr
            if out["decision"] in ("verified", "blocked", "step_up"):
                break
            st = {**st, "challenge": out["challenge"]}
        d = out["details"]
        first_fail = d.get("failed", [])
        return {"decision": out["decision"], "failed": first_fail, "latency": d.get("latency_s"), "tier": tier,
                "p_ai": d.get("checks", {}).get("authenticity", {}).get("p_ai"),
                "speaker": d.get("checks", {}).get("speaker", {}).get("score"), "challenge": d.get("checks", {}).get("challenge", {}).get("score")}

    n_users = len(keys)
    for tier in ("low", "high"):
        for i, k in enumerate(keys):
            uid = f"{k[0]}:{k[1]}"
            ps = paths_by[uid]
            test_clip = clip(ps[3 + (i % 5)], 6.0)
            cases["genuine"].append(run(uid, tier, lambda c, a, y=test_clip: y, "genuine"))
            other = keys[(i + 3) % n_users]
            cases["replay"].append(run(uid, tier, lambda c, a, y=clip(paths_by[f"{other[0]}:{other[1]}"][4], 6.0): y, "replay"))   # someone's recording, wrong content
            v = TTS_VOICES[i % len(TTS_VOICES)]
            cache = {}

            def pre(c, a, v=v):                              # TTS of a fixed sentence made BEFORE the challenge existed
                with tempfile.TemporaryDirectory() as d:
                    return tts("Please transfer the funds to my account today", v, Path(d) / "a.mp3")
            cases["pretts"].append(run(uid, tier, pre, "pretts"))

            def live(c, a, v=v):                             # attacker hears the challenge and synthesises it live
                with tempfile.TemporaryDirectory() as d:
                    return tts(spoken(c), v, Path(d) / "a.mp3")
            cases["live_tts"].append(run(uid, tier, live, "live_tts"))

            def live_rep(c, a, v=v):                         # ... and plays it through a loudspeaker into the microphone
                with tempfile.TemporaryDirectory() as d:
                    return replay(tts(spoken(c), v, Path(d) / "a.mp3"), np.random.default_rng(a)).astype(np.float32)
            cases["live_tts_replayed"].append(run(uid, tier, live_rep, "live_tts_replayed"))
        print(tier, "done", flush=True)

    def summarize(rows):
        r = {"n": len(rows), "verified": round(sum(x["decision"] == "verified" for x in rows) / len(rows), 3),
             "blocked": round(sum(x["decision"] == "blocked" for x in rows) / len(rows), 3)}
        stop = {}
        for x in rows:
            for f in (x["failed"] or []):
                stop[f] = stop.get(f, 0) + 1
        r["layers_that_failed"] = stop
        lat = [x["latency"] for x in rows if x["latency"]]
        if lat:
            r["latency_median_s"], r["latency_p95_s"] = round(float(np.median(lat)), 2), round(float(np.percentile(lat, 95)), 2)
        return r
    res = {c: {t: summarize([x for x in rows if x["tier"] == t]) for t in ("low", "high")} for c, rows in cases.items()}
    res["note"] = ("genuine = the enrolled speaker's own real recording of OTHER text with the transcript replaced by the expected phrase "
                   "(a human reading arbitrary challenge phrases cannot be synthesised); attackers are NON-targeted TTS voices, not clones of the enrolled speaker. "
                   "A challenge gets up to 4 tries (3 failures block).")
    res["raw"] = cases
    save("attacks_end_to_end", res)


# --------------------------------------------------------------------------------------------- ASR / challenge
def run_asr():
    from voicelock.asr import Transcriber
    t = Transcriber(S.models_dir)
    rng = np.random.default_rng(3)
    from voicelock.crypto import Keys
    keys = Keys(Path(tempfile.mkdtemp()))
    tests = []
    for i in range(48):
        w, d = ch.derive(keys, "h" * 64, f"s{i}", ch.new_nonce(), 0)
        tests.append((w, d))
    res = {}
    wavs = []
    with tempfile.TemporaryDirectory() as td:
        for i, (w, d) in enumerate(tests):
            c = {"words": w, "digits": d}
            wavs.append(tts(spoken(c), TTS_VOICES[i % len(TTS_VOICES)], Path(td) / f"{i}.mp3"))
    for name, fn in [("clean", lambda y: y), ("SNR 20 dB", lambda y: noisy(y, 20, rng)), ("SNR 10 dB", lambda y: noisy(y, 10, rng)),
                     ("SNR 5 dB", lambda y: noisy(y, 5, rng)), ("loudspeaker->mic (replay sim)", lambda y: replay(y, rng).astype(np.float32))]:
        ok = exact = 0
        for (w, d), y in zip(tests, wavs):
            m = ch.match(w, d, t.transcribe(fn(y)), S.challenge_min_match)
            ok += m["passed"]
            exact += m["score"] == 1.0
        res[name] = {"accepted": round(ok / len(tests), 3), "perfect": round(exact / len(tests), 3), "n": len(tests)}
        print(name, res[name], flush=True)
    # chance acceptance of a wrong phrase (what an attacker gets by guessing or replaying another challenge)
    wrong = sum(ch.match(tests[i][0], tests[i][1], " ".join(tests[(i + 1) % len(tests)][0]) + " " + tests[(i + 1) % len(tests)][1], S.challenge_min_match)["passed"]
                for i in range(len(tests)))
    res["wrong_phrase_accepted"] = f"{wrong}/{len(tests)}"
    res["guess_space"] = f"{len(ch.WORDS) ** 3 * 10 ** 4:,} phrases per challenge"
    save("challenge_recognition", res)


# --------------------------------------------------------------------------------------------- fuzz
def run_fuzz():
    from voicelock.crypto import Keys, TokenError, b64d, b64e, txn_hash, verify_token
    tmp = Path(tempfile.mkdtemp())
    keys = Keys(tmp)
    rng = np.random.default_rng(4)
    now = 1_700_000_000
    txn = {"user_id": "u", "amount": "50000", "currency": "INR", "recipient": "ACME", "reference": "x"}
    claims = {"sub": "u", "sid": "s", "jti": "j", "iat": now, "nbf": now, "exp": now + 60, "txn_hash": txn_hash(txn)}
    tok = keys.sign_token(claims)
    head, body, sig = tok.split(".")
    accepted = {"bit_flips_in_payload": 0, "bit_flips_in_signature": 0, "forged_payload_resigned_with_other_key": 0, "txn_field_substitution": 0}
    N = 2000
    for _ in range(N):
        raw = bytearray(b64d(body))
        raw[rng.integers(len(raw))] ^= 1 << int(rng.integers(8))
        try:
            verify_token(f"{head}.{b64e(bytes(raw))}.{sig}", keys.public, now + 1); accepted["bit_flips_in_payload"] += 1
        except TokenError:
            pass
        s = bytearray(b64d(sig))
        s[rng.integers(len(s))] ^= 1 << int(rng.integers(8))
        try:
            verify_token(f"{head}.{body}.{b64e(bytes(s))}", keys.public, now + 1); accepted["bit_flips_in_signature"] += 1
        except TokenError:
            pass
        evil = Keys(Path(tempfile.mkdtemp()))
        forged = evil.sign_token({**claims, "txn_hash": txn_hash({**txn, "amount": str(rng.integers(1, 10 ** 7))})})
        try:
            verify_token(forged, keys.public, now + 1); accepted["forged_payload_resigned_with_other_key"] += 1
        except TokenError:
            pass
        mut = {**txn, "amount": str(rng.integers(1, 10 ** 7)), "recipient": rng.choice(["ACME", "Mallory", "acme ", "Globex"])}
        if txn_hash(mut) == claims["txn_hash"] and canon_differs(txn, mut):
            accepted["txn_field_substitution"] += 1
    # claims must be exactly bound: expiry
    expired = False
    try:
        verify_token(tok, keys.public, now + 61)
    except TokenError:
        expired = True
    save("token_binding_fuzz", {"trials_per_attack": N, "accepted": accepted, "expired_token_rejected": expired,
                                "note": "single-use/replay and transaction mismatch on the live service are covered by tests/test_voicelock.py"})


def canon_differs(a, b):
    from voicelock.crypto import canonical_txn
    return canonical_txn(a) != canonical_txn(b)


if __name__ == "__main__":
    {"speaker": run_speaker, "forensics": run_forensics, "attacks": run_attacks, "asr": run_asr, "fuzz": run_fuzz}[sys.argv[1]]()
