"""VoiceLock settings. Everything can be overridden with VOICELOCK_* environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _f(name, default):
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("VOICELOCK_DATA", ROOT / "data" / "voicelock")))
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("VOICELOCK_MODELS", ROOT / "models" / "voicelock")))
    api_key: str = os.environ.get("VOICELOCK_API_KEY", "")     # empty = no key required (local demo)
    # audit + strict authenticity limits from the conformal study; one committed file shared with the browser build
    conformal_path: Path = field(default_factory=lambda: Path(os.environ.get("VOICELOCK_CONFORMAL_FILE", ROOT / "web" / "vl" / "conformal.json")))

    # sessions / tokens
    challenge_ttl_s: float = _f("VOICELOCK_CHALLENGE_TTL", 90)
    session_ttl_s: float = _f("VOICELOCK_SESSION_TTL", 300)
    token_ttl_s: float = _f("VOICELOCK_TOKEN_TTL", 60)
    max_attempts: int = int(_f("VOICELOCK_MAX_ATTEMPTS", 3))
    max_audio_s: float = _f("VOICELOCK_MAX_AUDIO_S", 20)
    min_audio_s: float = _f("VOICELOCK_MIN_AUDIO_S", 1.5)
    min_enroll_samples: int = int(_f("VOICELOCK_MIN_ENROLL", 3))

    # challenge: 3 words + 4 digits are spoken; one recognition error is tolerated
    challenge_min_match: float = _f("VOICELOCK_CHALLENGE_MATCH", 6 / 7)

    # Layer 1 - authenticity: P(AI) from the detector. A response is rejected when it reaches the limit for its risk tier.
    ai_limit: dict = field(default_factory=lambda: {"low": 0.50, "medium": 0.45, "high": 0.35, "critical": 0.25})
    ai_severe: float = 0.90        # at/above this the response is blocked immediately

    # Layer 2 - speaker (ECAPA-TDNN cosine similarity to the enrolled voiceprint)
    speaker_min: dict = field(default_factory=lambda: {"low": 0.40, "medium": 0.45, "high": 0.50, "critical": 0.55})
    speaker_severe: float = 0.15   # below this the speaker is clearly someone else: block immediately
    consistency_min: float = _f("VOICELOCK_CONSISTENCY_MIN", 0.35)   # 2 s segments vs the whole response

    # Layer 4 - transaction risk tiers (in the transaction currency, INR by default) and passes required
    amount_tiers: tuple = (10_000, 100_000, 1_000_000)             # < 10k low, < 1L medium, < 10L high, otherwise critical
    required_passes: dict = field(default_factory=lambda: {"low": 1, "medium": 1, "high": 1, "critical": 2})
    new_recipient_bumps_tier: bool = True
    off_hours: tuple = (8, 20)                                     # local hours with no extra scrutiny
    velocity_window_s: float = 3600
    velocity_limit: int = 3

    @property
    def db_path(self) -> Path:
        return self.data_dir / "voicelock.sqlite3"
