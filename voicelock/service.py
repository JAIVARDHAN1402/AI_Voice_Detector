"""VoiceLock orchestration: enrolment, transaction sessions, challenge-response, signed authorisation tokens."""
import hashlib
import json
import secrets
import threading
import time

import numpy as np

from . import challenge as ch
from . import risk
from .config import Settings
from .crypto import Keys, TokenError, canonical_txn, txn_hash, verify_token
from .guarantee import Guarantee
from .store import Store

SR = 16000


class VoiceLockError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class VoiceLock:
    def __init__(self, settings: Settings | None = None, speaker=None, asr=None, forensics=None, store=None, keys=None,
                 clock=time.time):
        self.s = settings or Settings()
        self.s.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or Store(self.s.db_path)
        self.keys = keys or Keys(self.s.data_dir)
        self.now = clock
        if speaker is None:
            from .speaker import SpeakerVerifier
            speaker = SpeakerVerifier(self.s.models_dir)
        if asr is None:
            from .asr import Transcriber
            asr = Transcriber(self.s.models_dir)
        if forensics is None:
            from .forensics import Forensics
            forensics = Forensics()
        self.speaker, self.asr, self.forensics = speaker, asr, forensics
        self.guarantee = Guarantee(self.s)
        self.limits = self.guarantee.limits(self.s.ai_limit)       # tier -> P(AI) limit (conformal-calibrated when available)
        self._infer = threading.Lock()     # one model forward at a time (CPU); crypto/DB work stays concurrent

    # ------------------------------------------------------------------ helpers
    def _check_audio(self, wave):
        wave = np.asarray(wave, np.float32)
        dur = len(wave) / SR
        if not np.isfinite(wave).all():
            raise VoiceLockError("audio contains invalid samples")
        if dur < self.s.min_audio_s:
            raise VoiceLockError(f"audio too short ({dur:.1f}s); at least {self.s.min_audio_s:.1f}s of speech is needed")
        if dur > self.s.max_audio_s:
            raise VoiceLockError(f"audio too long ({dur:.1f}s); maximum is {self.s.max_audio_s:.0f}s")
        return wave

    @staticmethod
    def _audio_id(wave) -> str:
        return hashlib.sha256(np.asarray(wave, np.float32).tobytes()).hexdigest()[:16]

    def _public_challenge(self, sess: dict) -> dict:
        words, digits = ch.derive(self.keys, sess["txn_hash"], sess["id"], sess["challenge_nonce"], sess["passes"])
        return {"challenge_id": sess["challenge_id"], "phrase": ch.phrase(words, digits), "words": words, "digits": digits,
                "expires_at": sess["challenge_expires"], "ttl_s": self.s.challenge_ttl_s,
                "pass": sess["passes"] + 1, "passes_required": sess["required_passes"]}

    def _issue_challenge(self, sid: str) -> dict:
        now = self.now()
        self.store.update_session(sid, challenge_id=secrets.token_hex(6), challenge_nonce=ch.new_nonce(),
                                  challenge_expires=now + self.s.challenge_ttl_s, challenge_used=0, status="awaiting_response")
        return self._public_challenge(self.store.get_session(sid))

    # ------------------------------------------------------------------ enrolment
    def enrollment_status(self, user_id: str) -> dict:
        n = len(self.store.voiceprints(user_id))
        return {"user_id": user_id, "samples": n, "required": self.s.min_enroll_samples, "enrolled": n >= self.s.min_enroll_samples}

    def enroll(self, user_id: str, wave) -> dict:
        user_id = user_id.strip()
        if not user_id or len(user_id) > 64:
            raise VoiceLockError("invalid user id")
        wave = self._check_audio(wave)
        with self._infer:
            try:
                f = self.forensics.analyze(wave)
            except ValueError as e:                        # e.g. "No speech detected" from the detector
                raise VoiceLockError(str(e)) from e
            if f["p_ai"] >= self.s.ai_limit["low"]:        # never enrol a synthetic voice as the reference (a fixed gate, not a tier guarantee)
                self.store.log("enroll_rejected", None, user_id, reason="synthetic", p_ai=round(f["p_ai"], 3))
                raise VoiceLockError("this sample looks synthetic and cannot be used as an enrolment sample", 422)
            emb = self.speaker.embed(self.speaker.prepare(wave))
        self.store.ensure_user(user_id)
        n = self.store.add_voiceprint(user_id, emb)
        self.store.log("enroll", None, user_id, samples=n, audio=self._audio_id(wave), p_ai=round(f["p_ai"], 3))
        return self.enrollment_status(user_id)

    def reset_enrollment(self, user_id: str):
        self.store.clear_voiceprints(user_id)
        self.store.log("enroll_reset", None, user_id)

    # ------------------------------------------------------------------ transaction session
    def start_transaction(self, txn: dict) -> dict:
        try:
            canon = canonical_txn(txn)
        except (ValueError, KeyError) as e:
            raise VoiceLockError(f"invalid transaction: {e}") from e
        user = canon["user_id"]
        if not self.store.user_exists(user) or not self.enrollment_status(user)["enrolled"]:
            raise VoiceLockError("user is not enrolled (voiceprint missing)", 409)
        now = self.now()
        a = risk.assess(canon, self.s, self.store.known_recipients(user),
                        self.store.recent_sessions(user, now - self.s.velocity_window_s), now)
        sid = "vl_" + secrets.token_urlsafe(9)
        self.store.put_session({
            "id": sid, "user_id": user, "txn": json.dumps(canon, sort_keys=True), "txn_hash": txn_hash(canon), "tier": a["tier"],
            "required_passes": self.s.required_passes[a["tier"]], "passes": 0, "attempts": 0, "status": "awaiting_response",
            "created": now, "expires": now + self.s.session_ttl_s, "challenge_id": "", "challenge_nonce": "",
            "challenge_expires": 0, "challenge_used": 0, "challenge_hashes": "[]", "scores": "[]"})
        c = self._issue_challenge(sid)
        self.store.log("session_started", sid, user, tier=a["tier"], reasons=a["reasons"], txn_hash=txn_hash(canon),
                       amount=canon["amount"], currency=canon["currency"])
        return {"session_id": sid, "tier": a["tier"], "risk": a, "transaction": canon, "txn_hash": txn_hash(canon), "challenge": c,
                "thresholds": {"ai_limit": self.limits[a["tier"]], "guarantee": self.guarantee.describe(a["tier"]), "speaker_min": self.s.speaker_min[a["tier"]],
                               "passes_required": self.s.required_passes[a["tier"]]}}

    # ------------------------------------------------------------------ challenge response
    def respond(self, session_id: str, wave) -> dict:
        sess = self.store.get_session(session_id)
        if sess is None:
            raise VoiceLockError("unknown session", 404)
        user, now = sess["user_id"], self.now()
        if sess["status"] in ("verified", "blocked", "expired"):
            raise VoiceLockError(f"session already {sess['status']}", 409)
        if now > sess["expires"]:
            self.store.update_session(session_id, status="expired")
            self.store.log("session_expired", session_id, user)
            raise VoiceLockError("session expired", 410)

        wave = self._check_audio(wave)
        # single-use challenge: burn it before looking at the audio so it can never be answered twice
        with self.store.lock:
            fresh = self.store.get_session(session_id)
            if fresh["challenge_used"]:
                raise VoiceLockError("this challenge was already answered; request a new one", 409)
            self.store.update_session(session_id, challenge_used=1)
        if now > sess["challenge_expires"]:
            return self._fail(sess, "challenge expired", {"expired": True}, severe=[], audio=wave)

        words, digits = ch.derive(self.keys, sess["txn_hash"], sess["id"], sess["challenge_nonce"], sess["passes"])
        t0 = time.perf_counter()
        try:
            with self._infer:
                heard = self.asr.transcribe(wave)
                forensic = self.forensics.analyze(wave)
                speech = self.speaker.prepare(wave)
                emb = self.speaker.embed(speech)
                enrolled = self.store.voiceprints(user)
                spk = self.speaker.score(enrolled, emb)
                cons = self.speaker.consistency(speech, emb)
        except ValueError as e:                            # silent / unusable audio: a failed attempt, never a server error
            return self._fail(sess, f"unusable audio: {e}", {"failed": ["audio"], "unusable": str(e)}, severe=[], audio=wave)
        latency = round(time.perf_counter() - t0, 3)
        c = ch.match(words, digits, heard, self.s.challenge_min_match)
        verdict = risk.judge(sess["tier"], c, forensic, spk, cons, self.s, self.limits,
                             self.guarantee.describe(sess["tier"]) if self.guarantee.active else None)
        verdict["latency_s"] = latency
        verdict["speaker"] = {**spk, "consistency": cons}
        verdict["authenticity"] = {k: forensic[k] for k in ("p_ai", "verdict", "windows")}

        if not verdict["all_ok"]:
            return self._fail(sess, "checks failed: " + ", ".join(verdict["failed"]), verdict, severe=verdict["severe"], audio=wave)
        return self._success(sess, verdict, ch.challenge_hash(words, digits, sess["challenge_nonce"]), wave)

    def _fail(self, sess, reason, verdict, severe, audio):
        sid, user = sess["id"], sess["user_id"]
        fails = sess["attempts"] + 1
        self.store.update_session(sid, attempts=fails)
        out = {"session_id": sid, "tier": sess["tier"], "details": verdict, "reason": reason, "failed_attempts": fails}
        if severe or fails >= self.s.max_attempts:
            self.store.update_session(sid, status="blocked")
            why = severe or [f"{fails} failed attempts"]
            self.store.log("blocked", sid, user, reason=reason, severe=why, audio=self._audio_id(audio),
                           checks=verdict.get("checks"))
            self.store.log("security_incident", sid, user, summary="; ".join(why), txn_hash=sess["txn_hash"], tier=sess["tier"])
            return {**out, "decision": "blocked", "incident": True, "message": "Transaction blocked and a security incident was recorded."}
        c = self._issue_challenge(sid)
        self.store.log("retry", sid, user, reason=reason, audio=self._audio_id(audio), checks=verdict.get("checks"))
        return {**out, "decision": "retry", "challenge": c, "message": "Verification failed. Please answer the new challenge."}

    def _success(self, sess, verdict, c_hash, audio):
        sid, user = sess["id"], sess["user_id"]
        passes = sess["passes"] + 1
        hashes = json.loads(sess["challenge_hashes"]) + [c_hash]
        scores = json.loads(sess["scores"]) + [{"p_ai": verdict["checks"]["authenticity"]["p_ai"],
                                                "speaker": verdict["checks"]["speaker"]["score"], "trust": verdict["trust_score"]}]
        self.store.update_session(sid, passes=passes, challenge_hashes=json.dumps(hashes), scores=json.dumps(scores))
        out = {"session_id": sid, "tier": sess["tier"], "details": verdict, "passes": passes, "passes_required": sess["required_passes"]}
        if passes < sess["required_passes"]:
            sess2 = self.store.get_session(sid)
            c = self._issue_challenge(sess2["id"])
            self.store.log("pass", sid, user, pass_no=passes, audio=self._audio_id(audio), trust=verdict["trust_score"])
            return {**out, "decision": "step_up", "challenge": c,
                    "message": f"Pass {passes}/{sess['required_passes']} verified. This transaction needs one more independent challenge."}
        now = self.now()
        jti = secrets.token_urlsafe(12)
        canon = json.loads(sess["txn"])
        token = self.keys.sign_token({
            "iss": "voicelock", "sub": user, "sid": sid, "jti": jti, "iat": int(now), "nbf": int(now),
            "exp": int(now + self.s.token_ttl_s), "txn_hash": sess["txn_hash"],
            "txn": {k: canon[k] for k in ("amount", "currency", "recipient")}, "tier": sess["tier"],
            "challenges": hashes, "scores": scores, "decision": "verified"})
        self.store.update_session(sid, status="verified")
        self.store.log("verified", sid, user, token_jti=jti, trust=verdict["trust_score"], audio=self._audio_id(audio),
                       tier=sess["tier"], passes=passes)
        return {**out, "decision": "verified", "token": token, "token_expires_at": int(now + self.s.token_ttl_s),
                "message": "Verified. A single-use authorisation token was issued for this exact transaction."}

    # ------------------------------------------------------------------ tokens (what the payment system calls)
    def verify_token(self, token: str, txn: dict, consume: bool = True) -> dict:
        """Check signature, expiry, exact transaction binding and single use. `consume` spends the token atomically."""
        try:
            claims = verify_token(token, self.keys.public, self.now())
        except TokenError as e:
            self.store.log("token_rejected", None, None, reason=str(e))
            return {"valid": False, "reason": str(e)}
        try:
            h = txn_hash(txn)
        except (ValueError, KeyError):
            return {"valid": False, "reason": "invalid transaction"}
        if h != claims["txn_hash"]:
            self.store.log("token_rejected", claims["sid"], claims["sub"], reason="transaction mismatch")
            return {"valid": False, "reason": "token was issued for a different transaction"}
        if self.store.token_used(claims["jti"]):
            self.store.log("token_rejected", claims["sid"], claims["sub"], reason="replay")
            return {"valid": False, "reason": "token already used"}
        if consume:
            if not self.store.consume_token(claims["jti"]):
                self.store.log("token_rejected", claims["sid"], claims["sub"], reason="replay")
                return {"valid": False, "reason": "token already used"}
            self.store.add_known_recipient(claims["sub"], json.loads(self.store.get_session(claims["sid"])["txn"])["recipient"])
            self.store.log("token_consumed", claims["sid"], claims["sub"], jti=claims["jti"])
        return {"valid": True, "claims": claims, "consumed": consume}
