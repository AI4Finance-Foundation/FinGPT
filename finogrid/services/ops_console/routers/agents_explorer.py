"""
Agent activity explorer — KYA status, wallet activity, daily spend summary.
"""
import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from ..dependencies import get_db, require_ops_key
from .....database.models.agent_ledger import (
    AgentAccount, AgentKYA, AgentWallet, MicroTransaction, AgentLedgerEntry, MicroTxStatus
)

log = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_ops_key)])


@router.get("")
async def list_agents(
    kya_status: str | None = Query(None, description="unverified|pending|basic|enhanced"),
    chain: str | None = Query(None),
    limit: int = Query(50, le=200),
    db: AsyncSession = Depends(get_db),
):
    """List all agent accounts with summary stats."""
    from sqlalchemy import and_
    conditions = []
    if kya_status:
        conditions.append(AgentAccount.kya_status == kya_status)
    if chain:
        conditions.append(AgentAccount.chain == chain)

    result = await db.execute(
        select(AgentAccount)
        .where(and_(*conditions) if conditions else True)
        .order_by(desc(AgentAccount.created_at))
        .limit(limit)
    )
    agents = result.scalars().all()

    return {
        "total": len(agents),
        "agents": [
            {
                "agent_account_id": str(a.id),
                "name": a.name,
                "kya_status": a.kya_status,
                "chain": a.chain,
                "prefund_balance_usdc": str(a.prefund_balance_usdc),
                "reserved_balance_usdc": str(a.reserved_balance_usdc),
                "available_usdc": str(
                    float(a.prefund_balance_usdc) - float(a.reserved_balance_usdc)
                ),
                "status": a.status,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in agents
        ],
    }


@router.get("/{agent_account_id}")
async def get_agent_detail(
    agent_account_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Full detail view for a single agent: KYA record, wallets, recent transactions,
    daily spend summary.
    """
    import uuid
    try:
        uid = uuid.UUID(agent_account_id)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid agent_account_id")

    # Agent account
    agent_result = await db.execute(select(AgentAccount).where(AgentAccount.id == uid))
    agent = agent_result.scalar_one_or_none()
    if agent is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Agent not found")

    # KYA record
    kya_result = await db.execute(select(AgentKYA).where(AgentKYA.agent_account_id == uid))
    kya = kya_result.scalar_one_or_none()

    # Wallets
    wallets_result = await db.execute(
        select(AgentWallet).where(AgentWallet.agent_account_id == uid)
    )
    wallets = wallets_result.scalars().all()

    # Recent transactions (last 20)
    txs_result = await db.execute(
        select(MicroTransaction)
        .where(
            MicroTransaction.payer_wallet_id.in_([w.id for w in wallets])
        )
        .order_by(desc(MicroTransaction.created_at))
        .limit(20)
    )
    txs = txs_result.scalars().all()

    # Today's spend
    from datetime import datetime, timezone
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    spend_result = await db.execute(
        select(func.sum(MicroTransaction.amount_usdc)).where(
            MicroTransaction.payer_wallet_id.in_([w.id for w in wallets]),
            MicroTransaction.status.in_([MicroTxStatus.SETTLED_OFFCHAIN, MicroTxStatus.SETTLED_ONCHAIN]),
            MicroTransaction.settled_at >= today_start,
        )
    )
    today_spend = float(spend_result.scalar() or 0)

    return {
        "agent_account_id": str(agent.id),
        "name": agent.name,
        "chain": agent.chain,
        "kya_status": agent.kya_status,
        "prefund_balance_usdc": str(agent.prefund_balance_usdc),
        "reserved_balance_usdc": str(agent.reserved_balance_usdc),
        "today_spend_usdc": today_spend,
        "kya": {
            "status": kya.status if kya else "unverified",
            "validator_name": kya.validator_name if kya else None,
            "validator_ref": kya.validator_ref if kya else None,
            "validator_token_present": bool(kya and kya.validator_token),
            "validator_expires_at": kya.validator_expires_at.isoformat() if (kya and kya.validator_expires_at) else None,
        } if kya else None,
        "wallets": [
            {
                "wallet_id": str(w.id),
                "label": w.label,
                "wallet_address": w.wallet_address,
                "loop_type": w.loop_type,
                "status": w.status,
                "daily_spent_usdc": str(w.daily_spent_usdc),
                "max_daily_usdc": str(w.max_daily_usdc),
            }
            for w in wallets
        ],
        "recent_transactions": [
            {
                "tx_id": str(t.id),
                "idempotency_key": t.idempotency_key,
                "payee_address": t.payee_address,
                "amount_usdc": str(t.amount_usdc),
                "status": t.status,
                "loop_type": t.loop_type,
                "settled_at": t.settled_at.isoformat() if t.settled_at else None,
            }
            for t in txs
        ],
    }
