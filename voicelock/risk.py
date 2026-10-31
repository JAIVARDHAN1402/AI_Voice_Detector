"""Layer 4 - risk and authorisation engine.

Policy: a transaction is authorised only if *every* layer passes (authenticity AND identity AND challenge). A strong score in one
layer can never compensate for a failure in another. The transaction's risk tier tightens the thresholds and, for the highest
tier, requires two independent challenge-responses ("dual control" by voice).
"""
import time

import numpy as np

from .config import Settings

TIERS = ["low", "medium", "high", "critical"]
TIER_RISK = {"low": 0.15, "medium": 0.40, "high": 0.70, "critical": 0.95}


def assess(txn: dict, s: Settings, known_recipients: list, recent_sessions: int, now: float | None = None) -> dict:
    """Transaction risk tier from the amount, whether the recipient is new, request velocity and the time of day."""
    now = time.time() if now is None else now
    amount = float(txn["amount"])
    idx = sum(amount >= t for t in s.amount_tiers)
    reasons = [f"amount {txn['amount']} {txn['currency']}"]
    base = idx
    if s.new_recipient_bumps_tier and txn["recipient"] not in {r for r in known_recipients}:
        idx += 1
        reasons.append("recipient not previously authorised")
    if recent_sessions >= s.velocity_limit:
        idx += 1
        reasons.append(f"{recent_sessions} requests in the last hour")
    hour = time.localtime(now).tm_hour
    if base >= 1 and not (s.off_hours[0] <= hour < s.off_hours[1]):
        idx += 1
        reasons.append("outside normal hours")
    tier = TIERS[min(idx, len(TIERS) - 1)]
    return {"tier": tier, "base_tier": TIERS[min(base, 3)], "risk_score": TIER_RISK[tier], "reasons": reasons}


def judge(tier: str, challenge: dict, forensic: dict, speaker: dict, consistency: dict | None, s: Settings) -> dict:
    """Evaluate one challenge-response against the thresholds of the transaction's tier."""
    p_ai = forensic["p_ai"]
    anomaly = forensic["acoustic"]["anomaly_score"]
    ai_limit = max(0.05, s.ai_limit[tier] - 0.10 * anomaly)       # acoustic anomalies tighten the limit slightly
    sp_min = s.speaker_min[tier]
    sp = speaker["score"]

    checks = {
        "challenge": {"ok": bool(challenge["passed"]), "score": challenge["score"], "heard": challenge["heard"]},
        "authenticity": {"ok": p_ai < ai_limit, "p_ai": round(p_ai, 3), "limit": round(ai_limit, 3), "verdict": forensic["verdict"]},
        "speaker": {"ok": sp >= sp_min, "score": round(sp, 3), "min": sp_min},
        "consistency": {"ok": True if consistency is None else consistency["min"] >= s.consistency_min,
                        "min": None if consistency is None else round(consistency["min"], 3), "threshold": s.consistency_min},
    }
    severe = []
    if p_ai >= s.ai_severe:
        severe.append("audio is almost certainly synthetic")
    if sp < s.speaker_severe:
        severe.append("voice does not match the enrolled speaker at all")
    failed = [k for k, v in checks.items() if not v["ok"]]

    ident = float(np.clip((sp - 0.10) / 0.60, 0, 1))
    trust = float((max(0.0, 1 - p_ai) * ident * max(challenge["score"], 0.0)) ** (1 / 3)) * (1 - 0.5 * anomaly)
    return {"checks": checks, "failed": failed, "severe": severe, "all_ok": not failed, "trust_score": round(trust, 3),
            "acoustic": forensic["acoustic"]}
