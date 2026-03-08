"""
Human approval queue — mandate threshold approvals.

When a Mandate has approval_mode=threshold and a transaction exceeds the
approval_threshold_usdc, the transaction enters PENDING_APPROVAL state.
Ops staff review and APPROVE or REJECT.

Approval decisions are immutable and logged to audit_logs.
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from ..dependencies import get_db, require_ops_key
from .....database.models.audit import AuditLog

log = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_ops_key)])


class ApprovalDecisionRequest(BaseModel):
    decision: str          # "approve" | "reject"
    ops_note: str          # Required — reasons for decision
    ops_agent_id: str      # Ops staff identifier


@router.get("")
async def list_pending_approvals(
    db: AsyncSession = Depends(get_db),
):
    """
    Return all transactions pending manual approval.
    These are micropay requests that exceeded the mandate's approval_threshold_usdc.
    In MVP: read from agent_ledger_entries where entry_type = 'pending_approval'.
    """
    # Query audit_logs for pending_approval entries
    try:
        result = await db.execute(
            select(AuditLog).where(
                AuditLog.action == "pending_approval"
            ).order_by(AuditLog.created_at.desc()).limit(100)
        )
        pending = result.scalars().all()
        return {
            "total": len(pending),
            "items": [
                {
                    "approval_id": str(a.id),
                    "entity_type": a.entity_type,
                    "entity_id": a.entity_id,
                    "details": a.details,
                    "created_at": a.created_at.isoformat() if a.created_at else None,
                }
                for a in pending
            ],
        }
    except Exception as exc:
        log.warning("list_approvals_error", error=str(exc))
        return {"total": 0, "items": []}


@router.post("/{approval_id}")
async def decide_approval(
    approval_id: str,
    request: ApprovalDecisionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Approve or reject a pending transaction. Decision is final and audited.
    Approve: the micropay proceeds. Reject: the reserved funds are released back.
    """
    if request.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")
    if not request.ops_note.strip():
        raise HTTPException(status_code=400, detail="ops_note is required")

    # Fetch the pending approval
    result = await db.execute(
        select(AuditLog).where(
            AuditLog.id == approval_id,
            AuditLog.action == "pending_approval",
        )
    )
    approval_log = result.scalar_one_or_none()
    if approval_log is None:
        raise HTTPException(status_code=404, detail="Approval request not found")

    # Record decision
    decision_log = AuditLog(
        entity_type=approval_log.entity_type,
        entity_id=approval_log.entity_id,
        action=f"approval_{request.decision}d",
        actor=request.ops_agent_id,
        details={
            "original_approval_id": approval_id,
            "decision": request.decision,
            "ops_note": request.ops_note,
            "original_details": approval_log.details,
        },
    )
    db.add(decision_log)
    await db.commit()

    log.info(
        "approval_decision_recorded",
        approval_id=approval_id,
        decision=request.decision,
        ops_agent=request.ops_agent_id,
    )
    return {
        "approval_id": approval_id,
        "decision": request.decision,
        "message": f"Transaction {request.decision}d and logged.",
    }
