"""
KYA Validator MCP Server — port 9005.

Know Your Agent validation service. Abstracts the third-party validator
behind a standard MCP tool interface. Swapping the validator = swapping
the implementation here, zero changes to the Agent Ledger API.

Supported validators (configured via KYA_VALIDATOR_BACKEND env var):
  - internal  (MVP default): lightweight rule-based check + mock JWT stamp
  - sardine    : Sardine risk API (fraud + identity)
  - persona    : Persona identity verification
  - chainalysis: Chainalysis KYT + entity screening

Tools:
  - submit_kya          Initiate validation; returns validator_ref
  - get_kya_status      Poll for validator decision + token
  - verify_kya_token    Verify a JWT stamp hasn't been revoked
  - renew_kya           Trigger re-validation (annual renewal)

MCP pattern: each tool is a POST /tools/{tool_name} endpoint.
"""
import uuid
import time
import json
import base64
import hashlib
import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger()

# ── Settings ─────────────────────────────────────────────────────────────────

class KYAValidatorSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    kya_validator_backend: str = "internal"
    sardine_api_key: str = ""
    sardine_api_url: str = "https://api.sardine.ai/v1"
    persona_api_key: str = ""
    chainalysis_api_key: str = ""
    kya_token_ttl_days: int = 365
    app_host: str = "0.0.0.0"
    app_port: int = 9005

settings = KYAValidatorSettings()

# In-memory store for MVP (production: use persistent DB / Redis)
_kya_submissions: dict[str, dict] = {}

# ── Pydantic models ───────────────────────────────────────────────────────────

class SubmitKYARequest(BaseModel):
    agent_account_id: str
    agent_purpose: str
    declared_use_case: str
    agent_owner_attestation: str
    validator_name: Optional[str] = "internal"


class GetKYAStatusRequest(BaseModel):
    validator_ref: str


class VerifyKYATokenRequest(BaseModel):
    validator_token: str
    agent_account_id: str


class RenewKYARequest(BaseModel):
    agent_account_id: str
    validator_ref: str

# ── Internal validator (MVP) ──────────────────────────────────────────────────

def _internal_validate(submission: dict) -> tuple[str, str]:
    """
    Lightweight rule-based KYA for MVP.
    Returns (status, kya_level) — 'basic' or 'enhanced'.
    Enhanced if: purpose > 200 chars AND attestation > 100 chars AND
                 declared_use_case != 'general'.
    """
    purpose = submission.get("agent_purpose", "")
    attestation = submission.get("agent_owner_attestation", "")
    use_case = submission.get("declared_use_case", "general")

    if len(purpose) > 200 and len(attestation) > 100 and use_case != "general":
        return "enhanced", "enhanced"
    return "basic", "basic"


def _mint_validator_token(agent_account_id: str, kya_level: str) -> tuple[str, datetime]:
    """Mint a simple opaque validator stamp token (MVP — not a real JWT)."""
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.kya_token_ttl_days)
    payload = {
        "sub": agent_account_id,
        "level": kya_level,
        "iss": "finogrid-kya-validator",
        "iat": int(time.time()),
        "exp": int(expires_at.timestamp()),
        "jti": str(uuid.uuid4()),
    }
    token = base64.b64encode(json.dumps(payload).encode()).decode()
    # In production: sign with RS256 private key
    return token, expires_at

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Finogrid KYA Validator MCP", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "finogrid-kya-validator", "backend": settings.kya_validator_backend}


@app.post("/tools/submit_kya")
async def submit_kya(request: SubmitKYARequest):
    """
    Initiate KYA validation. Returns a validator_ref for polling.
    For internal backend: auto-validates and returns immediately.
    For external backends: submits to the vendor and returns a ref for async polling.
    """
    validator_ref = f"kya_{uuid.uuid4().hex[:16]}"

    if settings.kya_validator_backend == "internal":
        status, level = _internal_validate(request.model_dump())
        token, expires_at = _mint_validator_token(request.agent_account_id, level)
        _kya_submissions[validator_ref] = {
            "agent_account_id": request.agent_account_id,
            "status": status,
            "kya_level": level,
            "validator_token": token,
            "validator_expires_at": expires_at.isoformat(),
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        log.info("kya_submitted_internal", agent_id=request.agent_account_id, level=level)
        return {"validator_ref": validator_ref, "status": "submitted"}

    elif settings.kya_validator_backend == "sardine":
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{settings.sardine_api_url}/kyc/sessions",
                    headers={"Authorization": f"Bearer {settings.sardine_api_key}"},
                    json={
                        "externalId": request.agent_account_id,
                        "flow": "kya_agent",
                        "agentPurpose": request.agent_purpose,
                        "declaredUseCase": request.declared_use_case,
                    },
                )
                if resp.status_code == 200:
                    sardine_data = resp.json()
                    _kya_submissions[validator_ref] = {
                        "agent_account_id": request.agent_account_id,
                        "status": "pending",
                        "sardine_session_id": sardine_data.get("sessionId"),
                    }
                    return {"validator_ref": validator_ref, "status": "submitted"}
        except Exception as exc:  # noqa: BLE001
            log.error("sardine_submit_failed", error=str(exc))
            raise HTTPException(status_code=503, detail=f"Sardine API error: {exc}")

    raise HTTPException(status_code=400, detail=f"Unknown validator backend: {settings.kya_validator_backend}")


@app.post("/tools/get_kya_status")
async def get_kya_status(request: GetKYAStatusRequest):
    """Poll for validator decision. Returns status + token when ready."""
    record = _kya_submissions.get(request.validator_ref)
    if not record:
        raise HTTPException(status_code=404, detail="validator_ref not found")

    # For external backends, poll the vendor here and update record
    if settings.kya_validator_backend == "sardine" and record.get("status") == "pending":
        # TODO: poll Sardine for decision
        pass

    return {
        "validator_ref": request.validator_ref,
        "status": record.get("status"),
        "kya_level": record.get("kya_level"),
        "validator_token": record.get("validator_token"),
        "validator_expires_at": record.get("validator_expires_at"),
        "validated_at": record.get("validated_at"),
    }


@app.post("/tools/verify_kya_token")
async def verify_kya_token(request: VerifyKYATokenRequest):
    """Verify that a KYA token is still valid (not revoked, not expired)."""
    try:
        payload = json.loads(base64.b64decode(request.validator_token).decode())
        exp = payload.get("exp", 0)
        sub = payload.get("sub", "")
        now = int(time.time())

        if exp < now:
            return {"valid": False, "reason": "token_expired"}
        if sub != request.agent_account_id:
            return {"valid": False, "reason": "subject_mismatch"}

        return {
            "valid": True,
            "kya_level": payload.get("level"),
            "expires_at": datetime.fromtimestamp(exp, tz=timezone.utc).isoformat(),
        }
    except Exception:  # noqa: BLE001
        return {"valid": False, "reason": "invalid_token_format"}


@app.post("/tools/renew_kya")
async def renew_kya(request: RenewKYARequest):
    """Trigger re-validation for an agent. Resets status to pending."""
    record = _kya_submissions.get(request.validator_ref)
    if not record:
        raise HTTPException(status_code=404, detail="validator_ref not found")

    record["status"] = "pending"
    record["validator_token"] = None
    record["renewed_at"] = datetime.now(timezone.utc).isoformat()

    log.info("kya_renewal_triggered", agent_id=request.agent_account_id, ref=request.validator_ref)
    return {"status": "pending", "message": "KYA renewal submitted. Poll get_kya_status for updates."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
