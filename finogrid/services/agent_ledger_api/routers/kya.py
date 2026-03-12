"""KYA (Know Your Agent) submit and status endpoints."""
import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..schemas import KYASubmitRequest, KYAStatusResponse
from ..dependencies import get_db, get_current_agent_account
from ..config import settings
from .....database.models.agent_ledger import AgentAccount, AgentKYA, KYAStatus

log = structlog.get_logger()
router = APIRouter()


@router.post("/{agent_account_id}/kya", response_model=KYAStatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_kya(
    agent_account_id: str,
    request: KYASubmitRequest,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """
    Submit KYA for this agent. Forwards to KYA validator MCP server.
    Returns status=pending immediately; poll GET endpoint for final result.
    KYA is required before any outbound micropayments are permitted.
    """
    if str(agent.id) != agent_account_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Upsert AgentKYA record
    result = await db.execute(
        select(AgentKYA).where(AgentKYA.agent_account_id == agent.id)
    )
    kya = result.scalar_one_or_none()

    if kya is None:
        kya = AgentKYA(agent_account_id=agent.id)
        db.add(kya)

    kya.agent_purpose = request.agent_purpose
    kya.declared_use_case = request.declared_use_case
    kya.agent_owner_attestation = request.agent_owner_attestation
    kya.validator_name = request.validator_name or "internal"
    kya.status = KYAStatus.PENDING

    # Update denormalized status on AgentAccount
    agent.kya_status = KYAStatus.PENDING
    await db.commit()
    await db.refresh(kya)

    # Forward to KYA validator MCP server (best-effort; status polled async)
    validator_ref = None
    if settings.kya_enabled:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{settings.kya_validator_mcp_url}/tools/submit_kya",
                    json={
                        "agent_account_id": str(agent.id),
                        "agent_purpose": request.agent_purpose,
                        "declared_use_case": request.declared_use_case,
                        "agent_owner_attestation": request.agent_owner_attestation,
                        "validator_name": kya.validator_name,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    validator_ref = data.get("validator_ref")
                    if validator_ref:
                        kya.validator_ref = validator_ref
                        await db.commit()
        except Exception as exc:  # noqa: BLE001
            log.warning("kya_validator_mcp_unavailable", error=str(exc))

    log.info("kya_submitted", agent_id=agent_account_id, validator=kya.validator_name)
    return KYAStatusResponse(
        agent_account_id=agent.id,
        kya_status=kya.status,
        validator_name=kya.validator_name,
        validator_ref=kya.validator_ref,
        validator_token_present=bool(kya.validator_token),
        validator_expires_at=kya.validator_expires_at.isoformat() if kya.validator_expires_at else None,
        validated_at=kya.validated_at.isoformat() if kya.validated_at else None,
        message=(
            "KYA submitted and under review. "
            "Poll GET /v1/agent-accounts/{id}/kya for status updates. "
            "Outbound payments will be unblocked once status reaches 'basic'."
        ),
    )


@router.get("/{agent_account_id}/kya", response_model=KYAStatusResponse)
async def get_kya_status(
    agent_account_id: str,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """
    Return current KYA status for this agent. Also polls the validator MCP
    server to pull any updated stamp and refresh the local record.
    """
    if str(agent.id) != agent_account_id:
        raise HTTPException(status_code=403, detail="Access denied")

    result = await db.execute(
        select(AgentKYA).where(AgentKYA.agent_account_id == agent.id)
    )
    kya = result.scalar_one_or_none()

    if kya is None:
        return KYAStatusResponse(
            agent_account_id=agent.id,
            kya_status="unverified",
            validator_name=None,
            validator_ref=None,
            validator_token_present=False,
            validator_expires_at=None,
            validated_at=None,
            message="No KYA submitted. POST /v1/agent-accounts/{id}/kya to begin.",
        )

    # Poll validator for update if pending
    if kya.status == KYAStatus.PENDING and settings.kya_enabled and kya.validator_ref:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{settings.kya_validator_mcp_url}/tools/get_kya_status",
                    json={"validator_ref": kya.validator_ref},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    new_status = data.get("status")
                    if new_status in ("basic", "enhanced"):
                        kya.status = KYAStatus(new_status)
                        kya.validator_token = data.get("validator_token")
                        import datetime
                        if data.get("validator_expires_at"):
                            kya.validator_expires_at = datetime.datetime.fromisoformat(
                                data["validator_expires_at"]
                            )
                        kya.validated_at = datetime.datetime.utcnow()
                        agent.kya_status = kya.status
                        await db.commit()
                        log.info("kya_status_updated", agent_id=agent_account_id, status=new_status)
        except Exception as exc:  # noqa: BLE001
            log.warning("kya_validator_poll_failed", error=str(exc))

    status_messages = {
        "unverified": "Submit KYA to begin.",
        "pending": "Validation in progress. Check back shortly.",
        "basic": "KYA basic verified. Transactions up to $1/day enabled.",
        "enhanced": "KYA enhanced verified. Transactions up to $100/day enabled.",
    }
    return KYAStatusResponse(
        agent_account_id=agent.id,
        kya_status=kya.status,
        validator_name=kya.validator_name,
        validator_ref=kya.validator_ref,
        validator_token_present=bool(kya.validator_token),
        validator_expires_at=kya.validator_expires_at.isoformat() if kya.validator_expires_at else None,
        validated_at=kya.validated_at.isoformat() if kya.validated_at else None,
        message=status_messages.get(str(kya.status), ""),
    )
