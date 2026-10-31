"""Cryptographic binding: canonical transaction hash (SHA-256) and Ed25519-signed, single-use authorization tokens.

Token format (JWS compact, alg EdDSA):  base64url(header) . base64url(payload) . base64url(signature)
The payload contains the transaction hash, so a token minted for one transaction can never authorise another one.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature


class TokenError(ValueError):
    pass


def b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def normalize_recipient(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip()).casefold()


def canonical_txn(txn: dict) -> dict:
    """Normalised, order-independent view of the fields a transfer is bound to."""
    try:
        amount = Decimal(str(txn["amount"])).quantize(Decimal("0.01"))
    except (InvalidOperation, KeyError) as e:
        raise ValueError("invalid amount") from e
    if amount <= 0:
        raise ValueError("amount must be positive")
    return {
        "user_id": str(txn["user_id"]).strip(),
        "amount": format(amount, "f"),
        "currency": str(txn.get("currency", "INR")).strip().upper(),
        "recipient": normalize_recipient(txn["recipient"]),
        "reference": str(txn.get("reference", "")).strip(),
    }


def txn_hash(txn: dict) -> str:
    blob = json.dumps(canonical_txn(txn), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


class Keys:
    """Ed25519 signing key + HMAC secret for challenge derivation, persisted under the data directory."""

    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        kp, sp = data_dir / "ed25519_private.pem", data_dir / "challenge_secret.key"
        if kp.exists():
            self.private = serialization.load_pem_private_key(kp.read_bytes(), password=None)
        else:
            self.private = Ed25519PrivateKey.generate()
            kp.write_bytes(self.private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                      serialization.NoEncryption()))
            self._lock_down(kp)
        if sp.exists():
            self.secret = sp.read_bytes()
        else:
            self.secret = secrets.token_bytes(32)
            sp.write_bytes(self.secret)
            self._lock_down(sp)
        self.public = self.private.public_key()
        raw = self.public.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.key_id = hashlib.sha256(raw).hexdigest()[:16]

    @staticmethod
    def _lock_down(path: Path):
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass

    def public_pem(self) -> str:
        return self.public.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()

    def hmac(self, *parts: str) -> bytes:
        return hmac.new(self.secret, "\x1f".join(parts).encode(), hashlib.sha256).digest()

    def sign_token(self, claims: dict) -> str:
        header = {"alg": "EdDSA", "typ": "VLT", "kid": self.key_id}
        head = b64e(json.dumps(header, separators=(",", ":")).encode())
        body = b64e(json.dumps(claims, sort_keys=True, separators=(",", ":")).encode())
        sig = self.private.sign(f"{head}.{body}".encode())
        return f"{head}.{body}.{b64e(sig)}"


def verify_token(token: str, public_key: Ed25519PublicKey, now: float | None = None) -> dict:
    """Return the claims if the signature and time window are valid; raise TokenError otherwise."""
    try:
        head, body, sig = token.split(".")
        header = json.loads(b64d(head))
        if header.get("alg") != "EdDSA":
            raise TokenError("unsupported algorithm")
        public_key.verify(b64d(sig), f"{head}.{body}".encode())
        claims = json.loads(b64d(body))
    except TokenError:
        raise
    except (ValueError, InvalidSignature, KeyError) as e:
        raise TokenError("invalid token signature or format") from e
    now = time.time() if now is None else now
    if claims.get("exp", 0) <= now:
        raise TokenError("token expired")
    if claims.get("nbf", 0) > now + 5:
        raise TokenError("token not yet valid")
    return claims
