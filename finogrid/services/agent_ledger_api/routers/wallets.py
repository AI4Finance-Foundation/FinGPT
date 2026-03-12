"""Agent wallet registration endpoints."""
import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..schemas import AgentWalletCreateRequest, AgentWalletCreateResponse
from ..dependencies import get_db, get_current_agent_account
from ..config import settings
from .....database.models.agent_ledger import AgentAccount, AgentWallet, LoopType

log = structlog.get_logger()
router = APIRouter()


@router.post("/{agent_account_id}/wallets", response_model=AgentWalletCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_wallet(
    agent_account_id: str,
    request: AgentWalletCreateRequest,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """
    Register a sub-wallet under this agent account.
    loop_type is set by the AgentOwner and is IMMUTABLE after creation.
    closed-loop wallets require a PaymentIntent for every payment.
    """
    if str(agent.id) != agent_account_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Verify wallet address not already registered under this agent
    existing = await db.execute(
        select(AgentWallet).where(
            AgentWallet.agent_account_id == agent.id,
            AgentWallet.wallet_address == request.wallet_address.lower(),
        )
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=409,
            detail=f"Wallet address {request.wallet_address} already registered for this agent",
        )

    # Register wallet with wallet factory MCP (optional — creates on-chain record)
    wallet_factory_confirmed = False
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.wallet_factory_mcp_url}/tools/register_wallet",
                json={
                    "wallet_address": request.wallet_address.lower(),
                    "agent_account_id": str(agent.id),
                    "chain": agent.chain,
                    "loop_type": request.loop_type,
                },
            )
            wallet_factory_confirmed = resp.status_code == 200
    except Exception as exc:  # noqa: BLE001
        log.warning("wallet_factory_mcp_unavailable", error=str(exc))

    spending_rules = {
        "loop_type": request.loop_type,
        "max_per_tx_usdc": str(request.max_per_tx_usdc),
        "max_daily_usdc": str(request.max_daily_usdc),
        "allowed_counterparties": request.allowed_counterparties,
        "expires_at": request.expires_at.isoformat() if request.expires_at else None,
        "max_uses": request.max_uses,
    }

    wallet = AgentWallet(
        agent_account_id=agent.id,
        label=request.label,
        wallet_address=request.wallet_address.lower(),
        chain=agent.chain,
        loop_type=LoopType(request.loop_type),
        max_per_tx_usdc=request.max_per_tx_usdc,
        max_daily_usdc=request.max_daily_usdc,
        allowed_counterparties=request.allowed_counterparties,
        expires_at=request.expires_at,
        max_uses=request.max_uses,
    )
    db.add(wallet)
    await db.commit()
    await db.refresh(wallet)

    log.info(
        "agent_wallet_created",
        wallet_id=str(wallet.id),
        agent_id=agent_account_id,
        loop_type=request.loop_type,
        wallet_factory_confirmed=wallet_factory_confirmed,
    )
    return AgentWalletCreateResponse(
        wallet_id=wallet.id,
        wallet_address=wallet.wallet_address,
        loop_type=wallet.loop_type,
        spending_rules=spending_rules,
    )


@router.get("/{agent_account_id}/wallets")
async def list_wallets(
    agent_account_id: str,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """List all wallets for this agent account."""
    if str(agent.id) != agent_account_id:
        raise HTTPException(status_code=403, detail="Access denied")

    result = await db.execute(
        select(AgentWallet)
        .where(AgentWallet.agent_account_id == agent.id)
        .order_by(AgentWallet.created_at.desc())
    )
    wallets = result.scalars().all()

    return {
        "agent_account_id": agent_account_id,
        "wallets": [
            {
                "wallet_id": str(w.id),
                "label": w.label,
                "wallet_address": w.wallet_address,
                "loop_type": w.loop_type,
                "status": w.status,
                "max_per_tx_usdc": float(w.max_per_tx_usdc),
                "max_daily_usdc": float(w.max_daily_usdc),
                "daily_spent_usdc": float(w.daily_spent_usdc),
                "use_count": w.use_count,
                "max_uses": w.max_uses,
                "expires_at": w.expires_at.isoformat() if w.expires_at else None,
            }
            for w in wallets
        ],
    }
