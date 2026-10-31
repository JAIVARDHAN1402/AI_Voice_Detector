"""Layer 1b - what the authenticity limit is known to do on AI voices the detector has never seen.

web/vl/conformal.json (written by scripts/export_research_web.py; the browser build reads the same file) holds two things per risk tier:

    audit   measured behaviour of VoiceLock's default P(AI) limits on held-out corpora (leave-one-domain-out): the share of unseen
            generators whose AI voices are accepted, and the share of genuine speech rejected.
    tiers   "strict" limits from domain-robust conformal calibration: a response is rejected at P(AI) >= limit, chosen so that the
            AI-accepted rate of a held-out corpus stays <= alpha. They reject a lot of genuine speech (see the paper), so they are optional.

VOICELOCK_CONFORMAL = "audit" (default)  keep the default limits and report the audited error rates
                      "strict"           use the conformal limits as the authenticity gate
                      "0" / "off"        hand-set limits, no claims (also the fallback when the file is missing or unreadable)
"""
import json
import os
from pathlib import Path

from .config import Settings

TIERS = ("low", "medium", "high", "critical")
MODES = ("audit", "strict", "off")


def _mode() -> str:
    v = os.environ.get("VOICELOCK_CONFORMAL", "audit").strip().lower()
    return "off" if v in ("0", "off", "false", "no") else "strict" if v in ("1", "strict", "true", "yes") else "audit"


class Guarantee:
    def __init__(self, s: Settings, path: Path | None = None, mode: str | None = None):
        self.path = Path(path) if path else s.conformal_path
        self.mode = mode or _mode()
        self.cal = None
        if self.mode != "off" and self.path.exists():
            try:
                cal = json.loads(self.path.read_text(encoding="utf-8"))
                if all(t in cal["tiers"] and t in cal["audit"] for t in TIERS):
                    self.cal = cal
            except (OSError, ValueError, KeyError, TypeError):
                self.cal = None
        if self.cal is None:
            self.mode = "off"

    @property
    def active(self) -> bool:
        return self.cal is not None

    @property
    def strict(self) -> bool:
        return self.mode == "strict"

    def limits(self, fallback: dict) -> dict:
        """tier -> P(AI) limit: a response at/above it is rejected."""
        if not self.strict:
            return dict(fallback)
        return {t: float(self.cal["tiers"][t]["p_ai_limit"]) for t in TIERS}

    def describe(self, tier: str) -> dict:
        if not self.active:
            return {"source": "heuristic", "note": "hand-set limit, no formal error guarantee"}
        if self.strict:
            t = self.cal["tiers"][tier]
            return {"source": "conformal", "alpha": t["alpha"], "confidence": round(1 - self.cal["delta"], 3),
                    "generators": self.cal["generators_used"], "genuine_rejected_pooled": round(t["genuine_rejected_pooled"], 3)}
        a = self.cal["audit"][tier]
        return {"source": "audit", "generators": a["generators"], "mean_generator_miss": round(a["mean_generator_miss"], 3),
                "pct_generators_miss_over_10pct": round(a["pct_generators_miss_over_10pct"], 3),
                "genuine_rejected": round(a["genuine_rejected"], 3)}
