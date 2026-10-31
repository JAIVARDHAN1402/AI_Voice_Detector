"""SQLite persistence (users, voiceprints, sessions, used tokens) + a tamper-evident, hash-chained audit log.

SQLite keeps the demo dependency-free; the SQL is plain, so moving to PostgreSQL only means swapping the connection.
"""
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, created REAL, known_recipients TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS voiceprints(user_id TEXT, idx INTEGER, emb BLOB, created REAL, PRIMARY KEY(user_id, idx));
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY, user_id TEXT, txn TEXT, txn_hash TEXT, tier TEXT, required_passes INTEGER, passes INTEGER DEFAULT 0,
  attempts INTEGER DEFAULT 0, status TEXT, created REAL, expires REAL, challenge_id TEXT, challenge_nonce TEXT,
  challenge_expires REAL, challenge_used INTEGER DEFAULT 0, challenge_hashes TEXT DEFAULT '[]', scores TEXT DEFAULT '[]');
CREATE TABLE IF NOT EXISTS used_tokens(jti TEXT PRIMARY KEY, used_at REAL);
CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, type TEXT, session_id TEXT, user_id TEXT,
  data TEXT, prev_hash TEXT, hash TEXT);
"""
GENESIS = "0" * 64


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA)

    # ---- users / voiceprints -------------------------------------------------
    def ensure_user(self, user_id: str):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO users(id, created) VALUES(?,?)", (user_id, time.time()))

    def user_exists(self, user_id: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone() is not None

    def add_voiceprint(self, user_id: str, emb: np.ndarray) -> int:
        with self.lock:
            n = self.db.execute("SELECT COUNT(*) FROM voiceprints WHERE user_id=?", (user_id,)).fetchone()[0]
            self.db.execute("INSERT INTO voiceprints VALUES(?,?,?,?)", (user_id, n, emb.astype("<f4").tobytes(), time.time()))
            return n + 1

    def voiceprints(self, user_id: str) -> np.ndarray:
        with self.lock:
            rows = self.db.execute("SELECT emb FROM voiceprints WHERE user_id=? ORDER BY idx", (user_id,)).fetchall()
        return np.stack([np.frombuffer(r["emb"], "<f4") for r in rows]) if rows else np.zeros((0, 0), np.float32)

    def clear_voiceprints(self, user_id: str):
        with self.lock:
            self.db.execute("DELETE FROM voiceprints WHERE user_id=?", (user_id,))

    def known_recipients(self, user_id: str) -> list:
        with self.lock:
            r = self.db.execute("SELECT known_recipients FROM users WHERE id=?", (user_id,)).fetchone()
        return json.loads(r["known_recipients"]) if r else []

    def add_known_recipient(self, user_id: str, recipient: str):
        known = set(self.known_recipients(user_id)) | {recipient}
        with self.lock:
            self.db.execute("UPDATE users SET known_recipients=? WHERE id=?", (json.dumps(sorted(known)), user_id))

    # ---- sessions --------------------------------------------------------------
    def put_session(self, s: dict):
        cols = ",".join(s)
        with self.lock:
            self.db.execute(f"INSERT OR REPLACE INTO sessions({cols}) VALUES({','.join('?' * len(s))})", tuple(s.values()))

    def get_session(self, sid: str):
        with self.lock:
            r = self.db.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return dict(r) if r else None

    def update_session(self, sid: str, **kw):
        sets = ",".join(f"{k}=?" for k in kw)
        with self.lock:
            self.db.execute(f"UPDATE sessions SET {sets} WHERE id=?", (*kw.values(), sid))

    def recent_sessions(self, user_id: str, since: float) -> int:
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM sessions WHERE user_id=? AND created>=?", (user_id, since)).fetchone()[0]

    # ---- single-use tokens -----------------------------------------------------
    def consume_token(self, jti: str) -> bool:
        """Atomically mark a token as spent. False if it was already used."""
        with self.lock:
            try:
                self.db.execute("INSERT INTO used_tokens VALUES(?,?)", (jti, time.time()))
                return True
            except sqlite3.IntegrityError:
                return False

    def token_used(self, jti: str) -> bool:
        with self.lock:
            return self.db.execute("SELECT 1 FROM used_tokens WHERE jti=?", (jti,)).fetchone() is not None

    # ---- audit log (hash chain) --------------------------------------------------
    def log(self, type_: str, session_id: str | None = None, user_id: str | None = None, **data) -> dict:
        with self.lock:
            prev = self.db.execute("SELECT hash FROM events ORDER BY seq DESC LIMIT 1").fetchone()
            prev_hash = prev["hash"] if prev else GENESIS
            ts = round(time.time(), 3)
            payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
            h = hashlib.sha256("|".join([prev_hash, str(ts), type_, session_id or "", user_id or "", payload]).encode()).hexdigest()
            cur = self.db.execute("INSERT INTO events(ts,type,session_id,user_id,data,prev_hash,hash) VALUES(?,?,?,?,?,?,?)",
                                  (ts, type_, session_id, user_id, payload, prev_hash, h))
            return {"seq": cur.lastrowid, "ts": ts, "type": type_, "session_id": session_id, "user_id": user_id,
                    "data": json.loads(payload), "hash": h}

    def events(self, limit: int = 100, session_id: str | None = None):
        q, args = "SELECT * FROM events", []
        if session_id:
            q += " WHERE session_id=?"
            args.append(session_id)
        q += " ORDER BY seq DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            rows = self.db.execute(q, args).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]

    def verify_chain(self) -> dict:
        """Recompute every hash; any edited, deleted or reordered event breaks the chain."""
        with self.lock:
            rows = self.db.execute("SELECT * FROM events ORDER BY seq").fetchall()
        prev = GENESIS
        for r in rows:
            h = hashlib.sha256("|".join([prev, str(r["ts"]), r["type"], r["session_id"] or "", r["user_id"] or "", r["data"]]).encode()).hexdigest()
            if r["prev_hash"] != prev or r["hash"] != h:
                return {"valid": False, "events": len(rows), "broken_at": r["seq"]}
            prev = h
        return {"valid": True, "events": len(rows), "head": prev}
