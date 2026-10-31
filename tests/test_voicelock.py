"""Security-logic tests for VoiceLock. The three ML models are replaced by stubs so these run in seconds and test the
protocol (binding, single use, expiry, thresholds, audit chain), not the models."""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voicelock import challenge as ch  # noqa: E402
from voicelock import risk  # noqa: E402
from voicelock.api import create_app  # noqa: E402
from voicelock.config import Settings  # noqa: E402
from voicelock.crypto import Keys, canonical_txn, txn_hash  # noqa: E402
from voicelock.guarantee import Guarantee  # noqa: E402
from voicelock.service import VoiceLock, VoiceLockError  # noqa: E402
from voicelock.speaker import SpeakerVerifier  # noqa: E402
from voicelock.store import Store  # noqa: E402

SR = 16000
DIM = 16


class StubSpeaker(SpeakerVerifier):
    """Voice identity is encoded in the first sample: speaker k -> embedding e_k (+ small noise via `blend`)."""

    def __init__(self):
        pass

    def embed(self, wave):
        k = int(round(float(wave[0]) * 10))
        v = np.zeros(DIM, np.float32)
        v[k] = 1.0
        v[(k + 1) % DIM] = float(wave[1])           # wave[1] = how much it leaks into a neighbouring identity
        return v / np.linalg.norm(v)

    def prepare(self, wave):
        return wave

    def consistency(self, wave, emb_full, **kw):
        return None


class StubASR:
    text = ""

    def transcribe(self, wave):
        return self.text


class StubForensics:
    p_ai = 0.02
    anomaly = 0.0
    silent = False

    def analyze(self, wave):
        if self.silent:
            raise ValueError("No speech detected - the recording seems silent or too quiet.")
        return {"p_ai": self.p_ai, "verdict": "ai" if self.p_ai >= 0.5 else "real", "windows": [],
                "acoustic": {"anomaly_score": self.anomaly, "flags": []}}


def wave(spk=0, leak=0.0, dur=3.0):
    w = np.random.default_rng(1).normal(0, 0.01, int(dur * SR)).astype(np.float32)
    w[0], w[1] = spk / 10, leak
    return w


class Clock:
    t = 1_700_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture(autouse=True)
def _heuristic_limits_by_default(monkeypatch):
    """The suite exercises the protocol with the hand-set limits; calibrated limits have their own tests below."""
    monkeypatch.setenv("VOICELOCK_CONFORMAL", "0")


@pytest.fixture
def env(tmp_path):
    s = Settings(data_dir=tmp_path)
    clock, asr, fx = Clock(), StubASR(), StubForensics()
    vl = VoiceLock(s, speaker=StubSpeaker(), asr=asr, forensics=fx, store=Store(s.db_path), keys=Keys(tmp_path), clock=clock)
    for _ in range(3):
        vl.enroll("alice", wave(spk=0))
    return vl, asr, fx, clock


def txn(**kw):
    t = {"user_id": "alice", "amount": "5000", "currency": "INR", "recipient": "ACME Pvt Ltd", "reference": "inv-1"}
    t.update(kw)
    return t


def answer(vl, asr, started, w=None):
    asr.text = started["challenge"]["phrase"]
    return vl.respond(started["session_id"], w if w is not None else wave(spk=0))


def test_enrollment_needs_enough_samples_and_rejects_synthetic(tmp_path):
    s = Settings(data_dir=tmp_path)
    fx = StubForensics()
    vl = VoiceLock(s, speaker=StubSpeaker(), asr=StubASR(), forensics=fx, store=Store(s.db_path), keys=Keys(tmp_path))
    vl.enroll("bob", wave(1))
    assert not vl.enrollment_status("bob")["enrolled"]
    with pytest.raises(VoiceLockError):
        vl.start_transaction(txn(user_id="bob"))
    fx.p_ai = 0.95
    with pytest.raises(VoiceLockError) as e:
        vl.enroll("bob", wave(1))
    assert e.value.status == 422


def test_happy_path_token_is_bound_and_single_use(env):
    vl, asr, _, _ = env
    st = vl.start_transaction(txn())
    r = answer(vl, asr, st)
    assert r["decision"] == "verified" and r["token"]
    assert vl.verify_token(r["token"], txn(amount="5000.00"), consume=False)["valid"]          # same transaction, normalised
    assert not vl.verify_token(r["token"], txn(amount="500000"), consume=False)["valid"]       # different amount
    assert not vl.verify_token(r["token"], txn(recipient="Mallory"), consume=False)["valid"]   # different recipient
    assert vl.verify_token(r["token"], txn(), consume=True)["valid"]
    assert vl.verify_token(r["token"], txn(), consume=True)["reason"] == "token already used"  # replay


def test_token_tamper_and_expiry(env):
    vl, asr, _, clock = env
    r = answer(vl, asr, vl.start_transaction(txn()))
    h, b, s = r["token"].split(".")
    forged = ".".join([h, b[:-3] + ("AAA" if not b.endswith("AAA") else "BBB"), s])
    assert not vl.verify_token(forged, txn(), consume=False)["valid"]
    clock.t += vl.s.token_ttl_s + 1
    assert vl.verify_token(r["token"], txn(), consume=False)["reason"] == "token expired"


def test_token_from_another_key_is_rejected(env, tmp_path_factory):
    vl, asr, _, _ = env
    r = answer(vl, asr, vl.start_transaction(txn()))
    other = Keys(tmp_path_factory.mktemp("other"))
    from voicelock.crypto import TokenError, verify_token
    with pytest.raises(TokenError):
        verify_token(r["token"], other.public, vl.now())


def test_challenge_is_unique_per_session_and_transaction(env):
    vl, _, _, _ = env
    a, b = vl.start_transaction(txn()), vl.start_transaction(txn())
    c = vl.start_transaction(txn(amount="6000"))
    assert len({a["challenge"]["phrase"], b["challenge"]["phrase"], c["challenge"]["phrase"]}) == 3
    w, d = ch.derive(vl.keys, a["txn_hash"], a["session_id"], vl.store.get_session(a["session_id"])["challenge_nonce"], 0)
    assert ch.phrase(w, d) == a["challenge"]["phrase"]          # deterministic given the same inputs


def test_challenge_is_single_use_and_expires(env):
    vl, asr, _, clock = env
    st = vl.start_transaction(txn(amount="100"))
    clock.t += vl.s.challenge_ttl_s + 1
    r = answer(vl, asr, st)                                     # too late -> retry with a fresh challenge
    assert r["decision"] == "retry" and r["details"].get("expired")
    assert r["challenge"]["phrase"] != st["challenge"]["phrase"]
    r2 = answer(vl, asr, {**st, "challenge": r["challenge"]})
    assert r2["decision"] == "verified"
    with pytest.raises(VoiceLockError):                         # session finished
        vl.respond(st["session_id"], wave(0))


def test_wrong_phrase_retries_then_blocks_with_incident(env):
    vl, asr, _, _ = env
    st = vl.start_transaction(txn())
    asr.text = "completely different words 9 9 9 9"
    for i in range(vl.s.max_attempts - 1):
        r = vl.respond(st["session_id"], wave(0))
        assert r["decision"] == "retry"
    r = vl.respond(st["session_id"], wave(0))
    assert r["decision"] == "blocked" and r["incident"]
    assert any(e["type"] == "security_incident" for e in vl.store.events(20))


def test_replayed_old_challenge_audio_fails(env):
    """A recording of a previous successful answer says the OLD phrase, which does not match the new transaction's phrase."""
    vl, asr, _, _ = env
    old = vl.start_transaction(txn())
    answer(vl, asr, old)
    new = vl.start_transaction(txn())
    asr.text = old["challenge"]["phrase"]
    assert vl.respond(new["session_id"], wave(0))["decision"] == "retry"


def test_synthetic_audio_is_blocked_and_thresholds_tighten_with_risk(env):
    vl, asr, fx, _ = env
    fx.p_ai = 0.95
    r = answer(vl, asr, vl.start_transaction(txn()))
    assert r["decision"] == "blocked" and "synthetic" in r["details"]["severe"][0]
    fx.p_ai = 0.40                                              # passes for a small transfer ...
    assert answer(vl, asr, vl.start_transaction(txn(amount="500", recipient="ACME Pvt Ltd")))["decision"] in ("verified", "retry")
    low = vl.start_transaction(txn(amount="500"))
    assert low["thresholds"]["ai_limit"] > vl.start_transaction(txn(amount="2000000"))["thresholds"]["ai_limit"]


def test_wrong_speaker_is_blocked(env):
    vl, asr, _, _ = env
    r = answer(vl, asr, vl.start_transaction(txn()), wave(spk=5))      # orthogonal identity -> cosine ~0
    assert r["decision"] == "blocked" and "enrolled speaker" in r["details"]["severe"][0]


def test_borderline_speaker_passes_small_transfer_but_retries_critical_one(env):
    vl, asr, _, _ = env
    w = wave(spk=0, leak=1.6)                                                   # cosine to the enrolled voice is ~0.53
    assert answer(vl, asr, vl.start_transaction(txn(amount="500")), w)["decision"] == "verified"      # needs >= 0.40
    r = answer(vl, asr, vl.start_transaction(txn(amount="5000000")), w)                               # critical needs >= 0.55
    assert r["decision"] == "retry" and r["details"]["failed"] == ["speaker"]


def test_critical_transaction_needs_two_independent_passes(env):
    vl, asr, _, _ = env
    st = vl.start_transaction(txn(amount="5000000"))
    assert st["tier"] == "critical" and st["challenge"]["passes_required"] == 2
    r1 = answer(vl, asr, st)
    assert r1["decision"] == "step_up" and "token" not in r1
    assert r1["challenge"]["phrase"] != st["challenge"]["phrase"]
    r2 = answer(vl, asr, {**st, "challenge": r1["challenge"]})
    assert r2["decision"] == "verified" and len(vl.verify_token(r2["token"], txn(amount="5000000"), consume=False)["claims"]["challenges"]) == 2


def test_new_recipient_raises_tier_and_known_recipient_lowers_it(env):
    vl, asr, _, _ = env
    first = vl.start_transaction(txn(amount="5000"))
    assert "recipient not previously authorised" in first["risk"]["reasons"]
    r = answer(vl, asr, first)
    vl.verify_token(r["token"], txn(amount="5000"), consume=True)       # executing records the recipient as known
    second = vl.start_transaction(txn(amount="5000"))
    assert "recipient not previously authorised" not in second["risk"]["reasons"]


def test_audit_chain_detects_tampering(env):
    vl, asr, _, _ = env
    answer(vl, asr, vl.start_transaction(txn()))
    assert vl.store.verify_chain()["valid"]
    vl.store.db.execute("UPDATE events SET data='{\"x\":1}' WHERE seq=2")
    assert not vl.store.verify_chain()["valid"]


def test_canonical_transaction_hash_is_stable():
    a = txn_hash(txn(amount="5000", recipient="ACME   Pvt Ltd"))
    b = txn_hash(txn(amount="5000.00", recipient="acme pvt ltd"))
    assert a == b and a != txn_hash(txn(amount="5000.01"))
    with pytest.raises(ValueError):
        canonical_txn(txn(amount="-1"))


def test_challenge_matching_tolerates_one_error_only():
    words, digits = ["amber", "candle", "river"], "4827"
    ok = lambda t: ch.match(words, digits, t, 6 / 7)["passed"]          # noqa: E731
    assert ok("amber candle river 4 8 2 7")
    assert ok("Amber, candle, river. Four eight two seven")
    assert ok("amber candle river 4827")
    assert ok("amber candle river 4 8 2 1")                              # one digit wrong
    assert not ok("amber candle river 4 8 1 1")                          # two wrong
    assert not ok("hello there")


def test_http_api_end_to_end(env):
    from fastapi.testclient import TestClient
    vl, asr, _, _ = env
    c = TestClient(create_app(vl, warm=False))

    def wav(w):
        b = io.BytesIO()
        sf.write(b, w, SR, format="WAV")
        return b.getvalue()

    t = txn()
    st = c.post("/v1/transactions", json=t).json()
    asr.text = st["challenge"]["phrase"]
    r = c.post(f"/v1/sessions/{st['session_id']}/respond", files={"audio": ("a.wav", wav(wave(0)), "audio/wav")}).json()
    assert r["decision"] == "verified"
    assert c.post("/v1/demo/transfer", json={"token": "garbage", "transaction": t}).status_code == 403
    assert c.post("/v1/demo/transfer", json={"token": r["token"], "transaction": {**t, "amount": "9999"}}).status_code == 403
    ok = c.post("/v1/demo/transfer", json={"token": r["token"], "transaction": t})
    assert ok.status_code == 200 and ok.json()["status"] == "executed"
    assert c.post("/v1/demo/transfer", json={"token": r["token"], "transaction": t}).status_code == 403   # replay
    assert c.get("/v1/events/verify").json()["valid"]
    assert c.post("/v1/transactions", json={**t, "user_id": "nobody"}).status_code == 409


def test_silent_audio_is_a_failed_attempt_not_a_server_error(env):
    vl, asr, fx, _ = env
    fx.silent = True
    st = vl.start_transaction(txn())
    r = answer(vl, asr, st)
    assert r["decision"] == "retry" and r["details"]["failed"] == ["audio"]
    with pytest.raises(VoiceLockError):
        vl.enroll("alice", wave(0))


# ---------------------------------------------------------------------------------------------- audited / calibrated limits
def _calibration(tmp_path, limits):
    tiers = {t: {"alpha": a, "tau_logit": 0.0, "p_ai_limit": lim, "genuine_rejected_pooled": 0.1}
             for (t, a), lim in zip((("low", 0.30), ("medium", 0.20), ("high", 0.10), ("critical", 0.05)), limits)}
    audit = {t: {"p_ai_limit": 0.5, "generators": 131, "mean_generator_miss": 0.25, "pct_generators_miss_over_10pct": 0.6,
                 "genuine_rejected": 0.22} for t in tiers}
    (tmp_path / "conformal.json").write_text(json.dumps({"delta": 0.1, "generators_used": 120, "tiers": tiers, "audit": audit}))


def test_guarantee_modes_and_fallbacks(tmp_path, monkeypatch):
    s = Settings(data_dir=tmp_path, models_dir=tmp_path, conformal_path=tmp_path / "conformal.json")
    assert not Guarantee(s).active and Guarantee(s).limits(s.ai_limit) == s.ai_limit          # no file -> heuristic
    assert Guarantee(s).describe("low")["source"] == "heuristic"
    _calibration(tmp_path, (0.4, 0.3, 0.2, 0.1))
    monkeypatch.delenv("VOICELOCK_CONFORMAL", raising=False)                                   # default = audit
    g = Guarantee(s)
    assert g.active and not g.strict and g.limits(s.ai_limit) == s.ai_limit                    # audit keeps the default limits ...
    d = g.describe("critical")
    assert d["source"] == "audit" and d["mean_generator_miss"] == 0.25 and d["genuine_rejected"] == 0.22   # ... and reports their audit
    monkeypatch.setenv("VOICELOCK_CONFORMAL", "strict")
    g = Guarantee(s)
    assert g.strict and g.limits(s.ai_limit)["critical"] == 0.1 and g.describe("critical")["alpha"] == 0.05
    monkeypatch.setenv("VOICELOCK_CONFORMAL", "off")
    assert not Guarantee(s).active
    (tmp_path / "conformal.json").write_text("{not json")
    monkeypatch.setenv("VOICELOCK_CONFORMAL", "strict")
    g = Guarantee(s)
    assert not g.active and g.mode == "off"                                                    # a corrupt file never crashes the service


def test_strict_limits_gate_each_tier_and_report_the_guarantee(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICELOCK_CONFORMAL", "strict")
    _calibration(tmp_path, (0.40, 0.30, 0.20, 0.05))
    s = Settings(data_dir=tmp_path, models_dir=tmp_path, conformal_path=tmp_path / "conformal.json")
    asr, fx = StubASR(), StubForensics()
    vl = VoiceLock(s, speaker=StubSpeaker(), asr=asr, forensics=fx, store=Store(s.db_path), keys=Keys(tmp_path), clock=Clock())
    for _ in range(3):
        vl.enroll("alice", wave(spk=0))
    fx.p_ai = 0.10                                                  # fine for a small transfer ...
    low = vl.start_transaction(txn(amount="500"))
    assert low["thresholds"]["guarantee"]["source"] == "conformal"          # (a new recipient lifts the tier by one)
    assert low["thresholds"]["guarantee"]["alpha"] == {"low": 0.30, "medium": 0.20}[low["tier"]]
    r = answer(vl, asr, low)
    assert r["decision"] == "verified" and r["details"]["checks"]["authenticity"]["guarantee"]["confidence"] == 0.9
    crit = vl.start_transaction(txn(amount="2000000"))              # ... but above the critical limit (0.05)
    r = answer(vl, asr, crit)
    assert r["decision"] in ("retry", "blocked") and "authenticity" in r["details"]["failed"]


def test_audit_mode_keeps_default_limits_and_shows_the_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICELOCK_CONFORMAL", "audit")
    _calibration(tmp_path, (0.001, 0.001, 0.001, 0.001))            # strict limits would reject everything; audit must ignore them
    s = Settings(data_dir=tmp_path, models_dir=tmp_path, conformal_path=tmp_path / "conformal.json")
    asr, fx = StubASR(), StubForensics()
    vl = VoiceLock(s, speaker=StubSpeaker(), asr=asr, forensics=fx, store=Store(s.db_path), keys=Keys(tmp_path), clock=Clock())
    for _ in range(3):
        vl.enroll("alice", wave(spk=0))
    fx.p_ai = 0.10
    started = vl.start_transaction(txn(amount="500"))
    assert started["thresholds"]["ai_limit"] == s.ai_limit[started["tier"]]
    assert started["thresholds"]["guarantee"]["source"] == "audit"
    r = answer(vl, asr, started)
    assert r["decision"] == "verified" and r["details"]["checks"]["authenticity"]["guarantee"]["source"] == "audit"


def test_acoustic_anomaly_only_tightens_a_strict_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICELOCK_CONFORMAL", "strict")
    _calibration(tmp_path, (0.40, 0.30, 0.20, 0.05))
    s = Settings(data_dir=tmp_path, models_dir=tmp_path, conformal_path=tmp_path / "conformal.json")
    g = Guarantee(s)
    fx = {"p_ai": 0.1, "verdict": "real", "windows": [], "acoustic": {"anomaly_score": 0.0, "flags": []}}
    sp = {"score": 0.9}
    ch_ = {"passed": True, "score": 1.0, "heard": ""}
    clean = risk.judge("high", ch_, fx, sp, None, s, g.limits(s.ai_limit), g.describe("high"))
    fx["acoustic"]["anomaly_score"] = 0.6
    noisy = risk.judge("high", ch_, fx, sp, None, s, g.limits(s.ai_limit), g.describe("high"))
    assert noisy["checks"]["authenticity"]["limit"] < clean["checks"]["authenticity"]["limit"] <= 0.20
