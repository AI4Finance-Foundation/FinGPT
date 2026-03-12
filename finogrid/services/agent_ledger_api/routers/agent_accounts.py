"""Agent account registration and balance endpoints."""
import secrets
import hashlib
import structlog
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..schemas import (
    AgentAccountCreateRequest, AgentAccountCreateResponse, AgentBalanceResponse
)
from ..dependencies import get_db, get_current_agent_account
from .....database.models.agent_ledger import AgentAccount, AgentLedgerEntry

log = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=AgentAccountCreateResponse, status_code=status.HTTP_201_CREATED)
async def register_agent_account(
    request: AgentAccountCreateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new master agent. No KYB required — instant provisioning.
    Returns the API key ONCE; store it securely.
    KYA must be submitted before outbound payments are permitted.
    """
    api_key = secrets.token_urlsafe(48)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    agent = AgentAccount(
        name=request.name,
        owner_client_id=request.owner_client_id,
        api_key_hash=key_hash,
        chain=request.chain,
        metadata_=request.metadata,
    )
    db.add(agent)
    await db.commit()
    await db.refresh(agent)

    log.info("agent_account_registered", agent_id=str(agent.id), name=agent.name)
    return AgentAccountCreateResponse(
        agent_account_id=agent.id,
        api_key=api_key,
        chain=agent.chain,
    )


@router.get("/{agent_account_id}/balance", response_model=AgentBalanceResponse)
async def get_balance(
    agent_account_id: str,
    db: AsyncSession = Depends(get_db),
    agent=Depends(get_current_agent_account),
):
    """Return balance, reserved amount, KYA status, and last 20 ledger entries."""
    if str(agent.id) != agent_account_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Access denied")

    entries_result = await db.execute(
        select(AgentLedgerEntry)
        .where(AgentLedgerEntry.agent_account_id == agent.id)
        .order_by(AgentLedgerEntry.created_at.desc())
        .limit(20)
    )
    entries = entries_result.scalars().all()

    available = float(agent.prefund_balance_usdc) - float(agent.reserved_balance_usdc)
    return AgentBalanceResponse(
        agent_account_id=agent.id,
        kya_status=agent.kya_status,
        prefund_balance_usdc=agent.prefund_balance_usdc,
        reserved_balance_usdc=agent.reserved_balance_usdc,
        available_balance_usdc=max(available, 0),
        chain=agent.chain,
        recent_entries=[
            {
                "id": str(e.id),
                "type": e.entry_type,
                "amount": float(e.amount_usdc),
                "balance_after": float(e.balance_after),
                "description": e.description,
                "at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ],
    )
