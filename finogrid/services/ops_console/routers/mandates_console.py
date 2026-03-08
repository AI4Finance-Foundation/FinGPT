"""
Mandate lifecycle console — activate, suspend, revoke mandates.
Every action is recorded in mandate_events (append-only).
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from ..dependencies import get_db, require_ops_key
from .....database.models.mandate import Mandate, MandateEvent, MandateStatus, MandateEventType

log = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_ops_key)])


class MandateActionRequest(BaseModel):
    ops_note: str
    ops_agent_id: str


class MandateRevokeRequest(MandateActionRequest):
    revocation_reason: str


@router.get("")
async def list_mandates(
    status: str | None = None,
    agent_account_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List mandates with optional filters."""
    from sqlalchemy import and_
    import uuid

    conditions = []
    if status:
        try:
            conditions.append(Mandate.status == MandateStatus(status))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    if agent_account_id:
        try:
            conditions.append(Mandate.agent_account_id == uuid.UUID(agent_account_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid agent_account_id")

    result = await db.execute(
        select(Mandate)
        .where(and_(*conditions) if conditions else True)
        .order_by(Mandate.created_at.desc())
        .limit(100)
    )
    mandates = result.scalars().all()

    return {
        "total": len(mandates),
        "mandates": [_mandate_to_dict(m) for m in mandates],
    }


@router.post("/{mandate_id}/activate")
async def activate_mandate(
    mandate_id: str,
    request: MandateActionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Activate a draft mandate — grant authority to the agent."""
    mandate = await _load_mandate(mandate_id, db)
    if mandate.status != MandateStatus.DRAFT:
        raise HTTPException(status_code=409, detail=f"Cannot activate mandate with status '{mandate.status}'")

    from datetime import datetime, timezone
    prev_status = str(mandate.status)
    mandate.status = MandateStatus.ACTIVE
    mandate.activated_at = datetime.now(timezone.utc)
    _emit_event(db, mandate, MandateEventType.ACTIVATED, request.ops_agent_id, prev_status, "active", request.ops_note)
    await db.commit()
    log.info("mandate_activated", mandate_id=mandate_id, ops_agent=request.ops_agent_id)
    return {"mandate_id": mandate_id, "status": "active", "message": "Mandate activated"}


@router.post("/{mandate_id}/suspend")
async def suspend_mandate(
    mandate_id: str,
    request: MandateActionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Temporarily suspend an active mandate. Can be resumed later."""
    mandate = await _load_mandate(mandate_id, db)
    if mandate.status != MandateStatus.ACTIVE:
        raise HTTPException(status_code=409, detail=f"Cannot suspend mandate with status '{mandate.status}'")

    prev_status = str(mandate.status)
    mandate.status = MandateStatus.SUSPENDED
    _emit_event(db, mandate, MandateEventType.SUSPENDED, request.ops_agent_id, prev_status, "suspended", request.ops_note)
    await db.commit()
    log.info("mandate_suspended", mandate_id=mandate_id, ops_agent=request.ops_agent_id)
    return {"mandate_id": mandate_id, "status": "suspended"}


@router.post("/{mandate_id}/resume")
async def resume_mandate(
    mandate_id: str,
    request: MandateActionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Resume a suspended mandate."""
    mandate = await _load_mandate(mandate_id, db)
    if mandate.status != MandateStatus.SUSPENDED:
        raise HTTPException(status_code=409, detail=f"Cannot resume mandate with status '{mandate.status}'")

    prev_status = str(mandate.status)
    mandate.status = MandateStatus.ACTIVE
    _emit_event(db, mandate, MandateEventType.RESUMED, request.ops_agent_id, prev_status, "active", request.ops_note)
    await db.commit()
    log.info("mandate_resumed", mandate_id=mandate_id, ops_agent=request.ops_agent_id)
    return {"mandate_id": mandate_id, "status": "active"}


@router.post("/{mandate_id}/revoke")
async def revoke_mandate(
    mandate_id: str,
    request: MandateRevokeRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Permanently revoke a mandate. IRREVERSIBLE.
    Revocation is immediate and blocks all future transactions under this mandate.
    """
    mandate = await _load_mandate(mandate_id, db)
    if mandate.status == MandateStatus.REVOKED:
        raise HTTPException(status_code=409, detail="Mandate already revoked")

    from datetime import datetime, timezone
    prev_status = str(mandate.status)
    mandate.status = MandateStatus.REVOKED
    mandate.revoked_at = datetime.now(timezone.utc)
    mandate.revocation_reason = request.revocation_reason
    _emit_event(
        db, mandate, MandateEventType.REVOKED,
        request.ops_agent_id, prev_status, "revoked",
        f"{request.ops_note} | Reason: {request.revocation_reason}"
    )
    await db.commit()
    log.info("mandate_revoked", mandate_id=mandate_id, reason=request.revocation_reason)
    return {"mandate_id": mandate_id, "status": "revoked", "message": "Mandate permanently revoked"}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _load_mandate(mandate_id: str, db: AsyncSession) -> Mandate:
    import uuid
    try:
        uid = uuid.UUID(mandate_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid mandate_id")
    result = await db.execute(select(Mandate).where(Mandate.id == uid))
    mandate = result.scalar_one_or_none()
    if mandate is None:
        raise HTTPException(status_code=404, detail="Mandate not found")
    return mandate


def _emit_event(
    db: AsyncSession,
    mandate: Mandate,
    event_type: MandateEventType,
    actor: str,
    prev_status: str,
    new_status: str,
    note: str,
):
    event = MandateEvent(
        mandate_id=mandate.id,
        event_type=event_type,
        actor=actor,
        previous_status=prev_status,
        new_status=new_status,
        note=note,
    )
    db.add(event)


def _mandate_to_dict(m: Mandate) -> dict:
    return {
        "mandate_id": str(m.id),
        "principal_id": str(m.principal_id),
        "agent_account_id": str(m.agent_account_id),
        "status": m.status,
        "scope": m.scope,
        "approval_mode": m.approval_mode,
        "max_amount_per_tx_usdc": str(m.max_amount_per_tx_usdc) if m.max_amount_per_tx_usdc else None,
        "max_daily_usdc": str(m.max_daily_usdc) if m.max_daily_usdc else None,
        "approval_threshold_usdc": str(m.approval_threshold_usdc) if m.approval_threshold_usdc else None,
        "allowed_corridors": m.allowed_corridors or [],
        "allowed_chains": m.allowed_chains or [],
        "activated_at": m.activated_at.isoformat() if m.activated_at else None,
        "expires_at": m.expires_at.isoformat() if m.expires_at else None,
        "revoked_at": m.revoked_at.isoformat() if m.revoked_at else None,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }
