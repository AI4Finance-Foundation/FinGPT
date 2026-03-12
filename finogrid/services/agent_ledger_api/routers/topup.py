"""Top-up endpoint — link an on-chain USDC deposit to prefund balance."""
import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import TopUpRequest, TopUpResponse
from ..dependencies import get_db, get_current_agent_account
from ..config import settings
from .....database.models.agent_ledger import AgentAccount, AgentLedgerEntry

log = structlog.get_logger()
router = APIRouter()


@router.post("/{agent_account_id}/topup", response_model=TopUpResponse)
async def topup(
    agent_account_id: str,
    request: TopUpRequest,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """
    Submit a USDC deposit tx hash to be credited to this agent's prefund balance.
    The chain_watcher will verify the tx on Base and credit the balance.
    This endpoint records the intent and returns status=pending_confirmation.
    KYA is NOT required for top-ups — agents can fund before verification completes.
    """
    if str(agent.id) != agent_account_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # Normalise tx hash
    tx_hash = request.deposit_tx_hash.strip().lower()
    if not tx_hash.startswith("0x") or len(tx_hash) != 66:
        raise HTTPException(
            status_code=400,
            detail="deposit_tx_hash must be a valid 32-byte tx hash (0x + 64 hex chars)",
        )

    topup_status = "pending_confirmation"
    credited_amount = None

    if settings.chain_enabled:
        # Ask chain_watcher to verify immediately (best-effort synchronous path)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{settings.wallet_factory_mcp_url}/tools/check_tx_confirmed",
                    json={
                        "tx_hash": tx_hash,
                        "deposit_address": settings.agent_ledger_deposit_address,
                        "usdc_contract": settings.usdc_contract_address_base,
                        "chain": settings.chain,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("confirmed") and data.get("amount_usdc"):
                        from decimal import Decimal
                        credited_amount = Decimal(str(data["amount_usdc"]))
                        agent.prefund_balance_usdc = (
                            Decimal(str(agent.prefund_balance_usdc)) + credited_amount
                        )
                        topup_status = "credited"

                        ledger_entry = AgentLedgerEntry(
                            agent_account_id=agent.id,
                            entry_type="credit",
                            amount_usdc=credited_amount,
                            balance_after=agent.prefund_balance_usdc,
                            reserved_balance_after=agent.reserved_balance_usdc,
                            on_chain_tx_hash=tx_hash,
                            description=f"USDC deposit confirmed: {tx_hash[:18]}...",
                        )
                        db.add(ledger_entry)
                        await db.commit()
                        log.info(
                            "topup_credited",
                            agent_id=agent_account_id,
                            tx_hash=tx_hash,
                            amount=str(credited_amount),
                        )
        except Exception as exc:  # noqa: BLE001
            log.warning("chain_verification_failed", tx_hash=tx_hash, error=str(exc))
    else:
        log.info("topup_queued_chain_disabled", agent_id=agent_account_id, tx_hash=tx_hash)

    message_map = {
        "pending_confirmation": (
            "Deposit tx recorded. The chain_watcher will verify and credit your balance "
            "within ~60 seconds. Poll GET /v1/agent-accounts/{id}/balance to confirm."
        ),
        "credited": (
            f"Deposit confirmed and credited. "
            f"Amount: {float(credited_amount):.2f} USDC added to your prefund balance."
        ),
    }

    return TopUpResponse(
        agent_account_id=agent.id,
        deposit_tx_hash=tx_hash,
        status=topup_status,
        message=message_map[topup_status],
    )
