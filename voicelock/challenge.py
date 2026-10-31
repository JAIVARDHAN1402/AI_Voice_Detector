"""Layer 3 - transaction-bound challenge engine.

The phrase the speaker must say (3 words + 4 digits) is derived with HMAC-SHA256 from a server secret and
(transaction hash, session id, one-time nonce, pass index). It therefore
  * cannot be predicted or pre-recorded (the nonce is fresh for every challenge),
  * cannot be reused for another transaction or session (both are inside the HMAC input),
  * expires, and can be answered only once.
"""
import difflib
import hashlib
import re
import secrets

from .crypto import Keys

# Acoustically distinct, easy to transcribe, no homophones of digits
WORDS = """amber anchor apple arrow basket beacon bridge cactus candle canyon carpet castle cobalt copper coral crystal
dolphin eagle ember falcon forest garden glacier harbor hazel island jacket jungle kettle lantern lemon marble meadow
mirror nectar orchid oyster palace pepper pillow planet pocket rabbit river saddle silver spider summer temple thunder
tiger velvet violet walnut window winter yellow zebra anvil bamboo canvas desert engine feather gadget helmet""".split()

DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
               "seven": "7", "eight": "8", "nine": "9"}


def derive(keys: Keys, txn_hash: str, session_id: str, nonce: str, pass_index: int):
    """-> (words, digits) for this exact transaction/session/nonce/pass."""
    d = hashlib.sha256(keys.hmac("challenge", txn_hash, session_id, nonce, str(pass_index))).digest()
    words, i = [], 0
    stream = hashlib.shake_256(d).digest(64)
    while len(words) < 3:
        w = WORDS[int.from_bytes(stream[i:i + 2], "big") % len(WORDS)]
        if w not in words:
            words.append(w)
        i += 2
    digits = "".join(str(stream[40 + j] % 10) for j in range(4))
    return words, digits


def phrase(words, digits) -> str:
    return " ".join(words) + ", " + " ".join(digits)


def challenge_hash(words, digits, nonce: str) -> str:
    return hashlib.sha256(f"{' '.join(words)}|{digits}|{nonce}".encode()).hexdigest()


def new_nonce() -> str:
    return secrets.token_hex(16)


def _tokens(text: str):
    text = text.casefold().replace("-", " ")
    words, digits = [], []
    for t in re.findall(r"[a-z]+|\d+", text):
        if t.isdigit():
            digits.extend(t)
        elif t in DIGIT_WORDS:
            digits.append(DIGIT_WORDS[t])
        else:
            words.append(t)
    return words, "".join(digits)


def match(expected_words, expected_digits: str, transcript: str, min_match: float) -> dict:
    """Fuzzy-compare the transcript with the expected phrase. One recognition error is tolerated by default."""
    heard_words, heard_digits = _tokens(transcript)
    pool, hits = list(heard_words), 0
    for w in expected_words:
        best = max(pool, key=lambda h: difflib.SequenceMatcher(None, w, h).ratio(), default=None)
        if best and difflib.SequenceMatcher(None, w, best).ratio() >= 0.8:
            hits += 1
            pool.remove(best)
    sm = difflib.SequenceMatcher(None, expected_digits, heard_digits, autojunk=False)
    dig_hits = sum(b.size for b in sm.get_matching_blocks())
    total = len(expected_words) + len(expected_digits)
    score = (hits + dig_hits) / total
    return {"passed": score >= min_match - 1e-9, "score": round(score, 3), "words_hit": hits, "digits_hit": dig_hits,
            "heard": transcript.strip()}
