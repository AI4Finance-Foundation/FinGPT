"""
Ledger explorer — cross-entity ledger view with filtering and export.
"""
import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from ..dependencies import get_db, require_ops_key
from .....database.models.agent_ledger import AgentLedgerEntry

log = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_ops_key)])


@router.get("")
async def query_ledger(
    agent_account_id: str | None = Query(None),
    entry_type: str | None = Query(None, description="credit|debit|refund|fee|intent_reserve|intent_release"),
    since: str | None = Query(None, description="ISO 8601 datetime filter"),
    until: str | None = Query(None, description="ISO 8601 datetime filter"),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
):
    """
    Query agent ledger entries with flexible filters.
    Returns paginated results sorted by created_at desc.
    """
    from sqlalchemy import and_
    from datetime import datetime, timezone

    conditions = []

    if agent_account_id:
        from sqlalchemy.dialects.postgresql import UUID
        import uuid
        try:
            uid = uuid.UUID(agent_account_id)
            conditions.append(AgentLedgerEntry.agent_account_id == uid)
        except ValueError:
            pass

    if entry_type:
        conditions.append(AgentLedgerEntry.entry_type == entry_type)

    if since:
        try:
            dt = datetime.fromisoformat(since)
            conditions.append(AgentLedgerEntry.created_at >= dt)
        except ValueError:
            pass

    if until:
        try:
            dt = datetime.fromisoformat(until)
            conditions.append(AgentLedgerEntry.created_at <= dt)
        except ValueError:
            pass

    try:
        query = (
            select(AgentLedgerEntry)
            .where(and_(*conditions) if conditions else True)
            .order_by(desc(AgentLedgerEntry.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(query)
        entries = result.scalars().all()

        return {
            "total": len(entries),
            "offset": offset,
            "limit": limit,
            "entries": [
                {
                    "id": str(e.id),
                    "agent_account_id": str(e.agent_account_id),
                    "entry_type": e.entry_type,
                    "amount_usdc": str(e.amount_usdc),
                    "balance_after": str(e.balance_after),
                    "reserved_balance_after": str(e.reserved_balance_after),
                    "description": e.description,
                    "on_chain_tx_hash": e.on_chain_tx_hash,
                    "micro_tx_id": str(e.micro_tx_id) if e.micro_tx_id else None,
                    "payment_intent_id": str(e.payment_intent_id) if e.payment_intent_id else None,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in entries
            ],
        }
    except Exception as exc:
        log.error("ledger_query_error", error=str(exc))
        return {"total": 0, "offset": offset, "limit": limit, "entries": []}
