"""
Manual review queue — held tasks, blocked agents, expired intents.
Ops staff triage, add notes, escalate, or clear exceptions.
"""
import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..dependencies import get_db, require_ops_key
from .....database.models.batch import PayoutTask, TaskStatus
from .....database.models.agent_ledger import AgentAccount, KYAStatus, PaymentIntent, IntentStatus

log = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_ops_key)])


@router.get("")
async def list_exceptions(
    exception_type: str = Query("all", description="all | held_tasks | kya_blocked | expired_intents"),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """
    Return all items currently in an exception state, grouped by type.
    This is the primary triage view for the ops team.
    """
    items = []

    if exception_type in ("all", "held_tasks"):
        try:
            held = (await db.execute(
                select(PayoutTask).where(PayoutTask.status == TaskStatus.HELD).limit(limit)
            )).scalars().all()
            items += [{
                "exception_type": "held_task",
                "id": str(t.id),
                "batch_id": str(t.batch_id),
                "status": t.status,
                "corridor": t.corridor,
                "amount_usd": str(t.amount_usd) if hasattr(t, "amount_usd") else None,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            } for t in held]
        except Exception as exc:
            log.warning("exceptions_held_tasks_error", error=str(exc))

    if exception_type in ("all", "kya_blocked"):
        try:
            blocked_agents = (await db.execute(
                select(AgentAccount).where(
                    AgentAccount.kya_status.in_([KYAStatus.UNVERIFIED, KYAStatus.PENDING])
                ).limit(limit)
            )).scalars().all()
            items += [{
                "exception_type": "kya_blocked_agent",
                "id": str(a.id),
                "name": a.name,
                "kya_status": a.kya_status,
                "chain": a.chain,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            } for a in blocked_agents]
        except Exception as exc:
            log.warning("exceptions_kya_blocked_error", error=str(exc))

    if exception_type in ("all", "expired_intents"):
        try:
            expired = (await db.execute(
                select(PaymentIntent).where(
                    PaymentIntent.status == IntentStatus.EXPIRED
                ).order_by(PaymentIntent.created_at.desc()).limit(limit)
            )).scalars().all()
            items += [{
                "exception_type": "expired_intent",
                "id": str(i.id),
                "amount_usdc": str(i.amount_usdc),
                "intent_category": i.intent_category,
                "expires_at": i.expires_at.isoformat() if i.expires_at else None,
                "audit_note": i.audit_note,
                "created_at": i.created_at.isoformat() if i.created_at else None,
            } for i in expired]
        except Exception as exc:
            log.warning("exceptions_expired_intents_error", error=str(exc))

    return {
        "exception_type": exception_type,
        "total": len(items),
        "items": items,
    }
