"""
Unified search across all Finogrid entities.
One endpoint, one query — returns cross-entity results ranked by relevance.
Ops staff can search by: tx hash, wallet address, agent name, batch ID, intent ID, client name.
"""
import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, text

from ..dependencies import get_db, require_ops_key
from .....database.models.client import Client
from .....database.models.batch import Batch, PayoutTask
from .....database.models.agent_ledger import AgentAccount, AgentWallet, PaymentIntent, MicroTransaction

log = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_ops_key)])


@router.get("")
async def unified_search(
    q: str = Query(..., min_length=2, description="Search term: ID, address, name, tx hash"),
    limit: int = Query(20, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Cross-entity search. Returns matches from:
    - clients (by name or ID)
    - batches (by ID)
    - payout_tasks (by ID or partner_tx_id)
    - agent_accounts (by name or ID)
    - agent_wallets (by address)
    - payment_intents (by ID)
    - micro_transactions (by idempotency_key or on_chain_tx_hash)
    """
    q_lower = q.lower().strip()
    results = []

    # Clients
    try:
        clients = (await db.execute(
            select(Client).where(
                or_(
                    Client.name.ilike(f"%{q_lower}%"),
                    text(f"CAST(clients.id AS TEXT) ILIKE '%{q_lower}%'"),
                )
            ).limit(5)
        )).scalars().all()
        results += [{"type": "client", "id": str(c.id), "label": c.name, "status": c.status} for c in clients]
    except Exception as exc:
        log.warning("search_clients_error", error=str(exc))

    # Agent accounts
    try:
        agents = (await db.execute(
            select(AgentAccount).where(
                or_(
                    AgentAccount.name.ilike(f"%{q_lower}%"),
                    text(f"CAST(agent_accounts.id AS TEXT) ILIKE '%{q_lower}%'"),
                )
            ).limit(5)
        )).scalars().all()
        results += [{
            "type": "agent_account",
            "id": str(a.id),
            "label": a.name,
            "kya_status": a.kya_status,
            "chain": a.chain,
        } for a in agents]
    except Exception as exc:
        log.warning("search_agents_error", error=str(exc))

    # Wallets by address
    try:
        wallets = (await db.execute(
            select(AgentWallet).where(AgentWallet.wallet_address.ilike(f"%{q_lower}%")).limit(5)
        )).scalars().all()
        results += [{
            "type": "agent_wallet",
            "id": str(w.id),
            "label": w.wallet_address,
            "loop_type": w.loop_type,
            "status": w.status,
        } for w in wallets]
    except Exception as exc:
        log.warning("search_wallets_error", error=str(exc))

    # Micro-transactions by idempotency_key or tx_hash
    try:
        txs = (await db.execute(
            select(MicroTransaction).where(
                or_(
                    MicroTransaction.idempotency_key.ilike(f"%{q_lower}%"),
                    MicroTransaction.on_chain_tx_hash.ilike(f"%{q_lower}%"),
                    text(f"CAST(micro_transactions.id AS TEXT) ILIKE '%{q_lower}%'"),
                )
            ).limit(5)
        )).scalars().all()
        results += [{
            "type": "micro_transaction",
            "id": str(t.id),
            "label": t.idempotency_key,
            "amount_usdc": str(t.amount_usdc),
            "status": t.status,
        } for t in txs]
    except Exception as exc:
        log.warning("search_txs_error", error=str(exc))

    # Batches
    try:
        batches = (await db.execute(
            select(Batch).where(
                text(f"CAST(batches.id AS TEXT) ILIKE '%{q_lower}%'")
            ).limit(5)
        )).scalars().all()
        results += [{"type": "batch", "id": str(b.id), "label": str(b.id), "status": b.status} for b in batches]
    except Exception as exc:
        log.warning("search_batches_error", error=str(exc))

    return {
        "query": q,
        "total": len(results),
        "results": results[:limit],
    }
