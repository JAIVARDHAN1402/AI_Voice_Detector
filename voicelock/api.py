"""VoiceLock HTTP API (FastAPI) + the demo web UI.

  POST /v1/users/{id}/enroll        multipart audio  -> add an enrolment sample
  GET  /v1/users/{id}/enrollment    -> enrolment status
  POST /v1/transactions             json  -> session + transaction-bound challenge
  POST /v1/sessions/{id}/respond    multipart audio  -> verified (+token) | step_up | retry | blocked
  POST /v1/tokens/verify            json  -> what a payment system calls before executing
  POST /v1/demo/transfer            json  -> mock payment endpoint that only executes with a valid token
  GET  /v1/events, /v1/events/verify, /v1/keys/public
"""
import threading
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import VoiceLock, VoiceLockError

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD = 12 * 1024 * 1024


class TxnIn(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    amount: str
    currency: str = "INR"
    recipient: str = Field(min_length=1, max_length=120)
    reference: str = Field(default="", max_length=200)


class TokenIn(BaseModel):
    token: str
    transaction: TxnIn
    consume: bool = True


def create_app(vl: VoiceLock | None = None, warm: bool = True) -> FastAPI:
    app = FastAPI(title="VoiceLock", version="1.0", docs_url="/docs")
    holder = {"vl": vl}

    def svc() -> VoiceLock:
        if holder["vl"] is None:
            holder["vl"] = VoiceLock()
        return holder["vl"]

    def auth(x_api_key: str = Header(default="")):
        key = svc().s.api_key
        if key and x_api_key != key:
            raise HTTPException(401, "invalid API key")

    @app.exception_handler(VoiceLockError)
    async def _err(_: Request, e: VoiceLockError):
        return JSONResponse({"error": str(e)}, status_code=e.status)

    async def read_audio(file: UploadFile) -> np.ndarray:
        from voicedetect.audio import load_bytes
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise VoiceLockError("file too large", 413)
        try:
            return load_bytes(data, Path(file.filename or "").suffix or ".wav")
        except Exception as e:  # noqa: BLE001
            raise VoiceLockError("could not read this audio file") from e

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "key_id": svc().keys.key_id}

    @app.get("/v1/keys/public", dependencies=[Depends(auth)])
    def public_key():
        return {"alg": "EdDSA", "kid": svc().keys.key_id, "pem": svc().keys.public_pem()}

    @app.get("/v1/users/{user_id}/enrollment", dependencies=[Depends(auth)])
    def enrollment(user_id: str):
        return svc().enrollment_status(user_id)

    @app.post("/v1/users/{user_id}/enroll", dependencies=[Depends(auth)])
    async def enroll(user_id: str, audio: UploadFile = File(...)):
        return svc().enroll(user_id, await read_audio(audio))

    @app.delete("/v1/users/{user_id}/enrollment", dependencies=[Depends(auth)])
    def reset(user_id: str):
        svc().reset_enrollment(user_id)
        return svc().enrollment_status(user_id)

    @app.post("/v1/transactions", dependencies=[Depends(auth)])
    def start(txn: TxnIn):
        return svc().start_transaction(txn.model_dump())

    @app.post("/v1/sessions/{session_id}/respond", dependencies=[Depends(auth)])
    async def respond(session_id: str, audio: UploadFile = File(...)):
        return svc().respond(session_id, await read_audio(audio))

    @app.post("/v1/tokens/verify", dependencies=[Depends(auth)])
    def verify(body: TokenIn):
        return svc().verify_token(body.token, body.transaction.model_dump(), body.consume)

    @app.post("/v1/demo/transfer", dependencies=[Depends(auth)])
    def demo_transfer(body: TokenIn):
        r = svc().verify_token(body.token, body.transaction.model_dump(), consume=True)
        if not r["valid"]:
            svc().store.log("transfer_denied", None, body.transaction.user_id, reason=r["reason"])
            return JSONResponse({"status": "denied", "reason": r["reason"]}, status_code=403)
        t = body.transaction
        svc().store.log("transfer_executed", r["claims"]["sid"], t.user_id, amount=t.amount, currency=t.currency, recipient=t.recipient)
        return {"status": "executed", "amount": t.amount, "currency": t.currency.upper(), "recipient": t.recipient,
                "authorised_by": r["claims"]["sid"]}

    @app.get("/v1/events", dependencies=[Depends(auth)])
    def events(limit: int = 60, session_id: str | None = None):
        return {"events": svc().store.events(min(limit, 500), session_id)}

    @app.get("/v1/events/verify", dependencies=[Depends(auth)])
    def chain():
        return svc().store.verify_chain()

    if warm and vl is None:
        def _warm():                              # load the three models once so the first user request is not slow
            try:
                v = svc()
                v.forensics._detector()
                v.speaker._encoder()
                v.asr._get()
            except Exception:  # noqa: BLE001
                pass
        threading.Thread(target=_warm, daemon=True).start()

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="ui")
    return app
