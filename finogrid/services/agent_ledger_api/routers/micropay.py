"""Micropay endpoint — the core settlement path."""
import uuid
import structlog
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..schemas import MicroPayRequest, MicroPayResponse
from ..dependencies import get_db, get_current_agent_account, assert_kya_status, kya_daily_limit
from ..config import settings
from .....database.models.agent_ledger import (
    AgentAccount, AgentWallet, AgentKYA, PaymentIntent, MicroTransaction,
    AgentLedgerEntry, IntentStatus, LoopType, MicroTxStatus,
)

log = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=MicroPayResponse, status_code=status.HTTP_201_CREATED)
async def micropay(
    request: MicroPayRequest,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """
    Execute a micropayment. Full compliance gate sequence:
    1. KYA status ≥ basic + validator token not expired
    2. Idempotency check (return existing result if key seen before)
    3. Wallet ownership + status
    4. Closed-loop: PaymentIntent required + validation
    5. Counterparty allowlist
    6. Per-tx cap
    7. Daily velocity cap (KYA level limits + wallet limits)
    8. Wallet expiry + max_uses
    9. Available balance check
    10. Settle off-chain, emit ledger entry
    """
    # ── Gate 1: KYA ──────────────────────────────────────────────────────────
    if settings.kya_enabled:
        await assert_kya_status(agent, required_level="basic")

        # Check validator token expiry
        kya_result = await db.execute(
            select(AgentKYA).where(AgentKYA.agent_account_id == agent.id)
        )
        kya = kya_result.scalar_one_or_none()
        now = datetime.now(timezone.utc)
        if kya and kya.validator_expires_at and kya.validator_expires_at < now:
            raise HTTPException(
                status_code=403,
                detail=(
                    "KYA validator token has expired. "
                    "Re-submit KYA at POST /v1/agent-accounts/{id}/kya to renew."
                ),
            )

    # ── Gate 2: Idempotency ───────────────────────────────────────────────────
    existing_result = await db.execute(
        select(MicroTransaction).where(
            MicroTransaction.idempotency_key == request.idempotency_key
        )
    )
    existing_tx = existing_result.scalar_one_or_none()
    if existing_tx:
        # Return the cached result — safe to replay
        available = float(agent.prefund_balance_usdc) - float(agent.reserved_balance_usdc)
        return MicroPayResponse(
            transaction_id=existing_tx.id,
            idempotency_key=existing_tx.idempotency_key,
            status=existing_tx.status,
            amount_usdc=existing_tx.amount_usdc,
            loop_type=existing_tx.loop_type,
            payment_intent_id=existing_tx.payment_intent_id,
            payer_available_balance_after=Decimal(str(max(available, 0))),
            settled_at=existing_tx.settled_at.isoformat() if existing_tx.settled_at else "",
            on_chain_tx_hash=existing_tx.on_chain_tx_hash,
            message="Idempotent replay — original transaction returned.",
        )

    # ── Gate 3: Wallet ────────────────────────────────────────────────────────
    wallet_result = await db.execute(
        select(AgentWallet).where(
            AgentWallet.id == request.payer_wallet_id,
            AgentWallet.agent_account_id == agent.id,
        )
    )
    wallet = wallet_result.scalar_one_or_none()
    if wallet is None:
        raise HTTPException(status_code=404, detail="Wallet not found or access denied")
    if wallet.status != "active":
        raise HTTPException(status_code=400, detail=f"Wallet status is '{wallet.status}', not active")

    now = datetime.now(timezone.utc)

    # ── Gate 4: Closed-loop intent ────────────────────────────────────────────
    intent = None
    if wallet.loop_type == LoopType.CLOSED:
        if not request.payment_intent_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Wallet is closed-loop. payment_intent_id is required. "
                    "Create one at POST /v1/payment-intents first."
                ),
            )
        intent_result = await db.execute(
            select(PaymentIntent).where(PaymentIntent.id == request.payment_intent_id)
        )
        intent = intent_result.scalar_one_or_none()
        if intent is None:
            raise HTTPException(status_code=404, detail="PaymentIntent not found")
        if intent.payer_wallet_id != wallet.id:
            raise HTTPException(status_code=403, detail="PaymentIntent does not belong to this wallet")
        if intent.status != IntentStatus.RESERVED:
            raise HTTPException(
                status_code=409,
                detail=f"PaymentIntent status is '{intent.status}'. Must be 'reserved'.",
            )
        if intent.expires_at < now:
            raise HTTPException(
                status_code=410,
                detail="PaymentIntent has expired. Create a new intent or supersede it.",
            )
        if abs(float(intent.amount_usdc) - float(request.amount_usdc)) > 0.00001:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Amount mismatch: intent reserved {float(intent.amount_usdc):.8f} USDC, "
                    f"payment requests {float(request.amount_usdc):.8f} USDC."
                ),
            )

    # ── Gate 5: Counterparty allowlist ────────────────────────────────────────
    if wallet.allowed_counterparties:
        if request.payee_address.lower() not in [a.lower() for a in wallet.allowed_counterparties]:
            raise HTTPException(
                status_code=403,
                detail=f"Payee {request.payee_address} not in wallet's allowed_counterparties allowlist.",
            )

    # ── Gate 6: Per-tx cap ────────────────────────────────────────────────────
    if request.amount_usdc > wallet.max_per_tx_usdc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Amount {float(request.amount_usdc):.2f} USDC exceeds per-tx cap "
                f"{float(wallet.max_per_tx_usdc):.2f} USDC."
            ),
        )

    # ── Gate 7: Daily velocity (wallet + KYA level) ───────────────────────────
    # Reset daily counter if needed
    if wallet.daily_reset_at is None or wallet.daily_reset_at < now:
        wallet.daily_spent_usdc = Decimal("0")
        wallet.daily_reset_at = now + timedelta(hours=24)

    new_daily_total = Decimal(str(wallet.daily_spent_usdc)) + request.amount_usdc
    if new_daily_total > wallet.max_daily_usdc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Daily wallet limit exceeded. "
                f"Spent today: {float(wallet.daily_spent_usdc):.2f} USDC, "
                f"Limit: {float(wallet.max_daily_usdc):.2f} USDC."
            ),
        )

    # KYA-level daily aggregate across all wallets
    if settings.kya_enabled:
        kya_limit = Decimal(str(kya_daily_limit(str(agent.kya_status))))
        if kya_limit > 0:
            # Sum today's transactions across all wallets
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            daily_total_result = await db.execute(
                select(func.sum(MicroTransaction.amount_usdc)).where(
                    MicroTransaction.payer_wallet_id.in_(
                        select(AgentWallet.id).where(AgentWallet.agent_account_id == agent.id)
                    ),
                    MicroTransaction.status.in_([
                        MicroTxStatus.SETTLED_OFFCHAIN, MicroTxStatus.SETTLED_ONCHAIN
                    ]),
                    MicroTransaction.settled_at >= day_start,
                )
            )
            daily_total = daily_total_result.scalar() or Decimal("0")
            if daily_total + request.amount_usdc > kya_limit:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"KYA daily limit exceeded for level '{agent.kya_status}'. "
                        f"Today's total: {float(daily_total):.2f} USDC, "
                        f"KYA limit: {float(kya_limit):.2f} USDC. "
                        "Submit enhanced KYA for higher limits."
                    ),
                )

    # ── Gate 8: Wallet expiry + max_uses ─────────────────────────────────────
    if wallet.expires_at and wallet.expires_at < now:
        raise HTTPException(status_code=400, detail="Wallet has expired.")
    if wallet.max_uses is not None and wallet.use_count >= wallet.max_uses:
        raise HTTPException(status_code=400, detail="Wallet has reached max_uses limit.")

    # ── Gate 9: Available balance ─────────────────────────────────────────────
    # For open-loop: available = prefund - reserved_balance
    # For closed-loop: amount is already reserved in intent
    available = Decimal(str(agent.prefund_balance_usdc)) - Decimal(str(agent.reserved_balance_usdc))
    if wallet.loop_type == LoopType.OPEN and request.amount_usdc > available:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Insufficient available balance. "
                f"Available: {float(available):.2f} USDC, "
                f"Requested: {float(request.amount_usdc):.2f} USDC."
            ),
        )

    # ── Gate 10: Settle off-chain ─────────────────────────────────────────────
    settled_at = now

    micro_tx = MicroTransaction(
        idempotency_key=request.idempotency_key,
        payer_wallet_id=wallet.id,
        payee_address=request.payee_address.lower(),
        amount_usdc=request.amount_usdc,
        chain=agent.chain,
        loop_type=wallet.loop_type,
        payment_intent_id=intent.id if intent else None,
        x402_payment_header=request.x402_payment_header,
        x402_resource_url=request.x402_resource_url,
        status=MicroTxStatus.SETTLED_OFFCHAIN,
        settled_at=settled_at,
        metadata_=request.metadata,
    )
    db.add(micro_tx)
    await db.flush()

    # Update balances
    if wallet.loop_type == LoopType.CLOSED and intent:
        # Closed-loop: release reservation and debit prefund
        agent.reserved_balance_usdc = Decimal(str(agent.reserved_balance_usdc)) - request.amount_usdc
        intent.status = IntentStatus.CONSUMED
        intent.consumed_micro_tx_id = micro_tx.id
    # Both loop types: debit prefund
    agent.prefund_balance_usdc = Decimal(str(agent.prefund_balance_usdc)) - request.amount_usdc

    # Update wallet counters
    wallet.daily_spent_usdc = new_daily_total
    wallet.use_count = (wallet.use_count or 0) + 1

    balance_after = Decimal(str(agent.prefund_balance_usdc))
    reserved_after = Decimal(str(agent.reserved_balance_usdc))

    # Ledger entry
    ledger_entry = AgentLedgerEntry(
        agent_account_id=agent.id,
        entry_type="debit",
        amount_usdc=request.amount_usdc,
        balance_after=balance_after,
        reserved_balance_after=reserved_after,
        micro_tx_id=micro_tx.id,
        payment_intent_id=intent.id if intent else None,
        description=(
            f"Micropay to {request.payee_address[:10]}... "
            f"via {'closed-loop intent ' + str(intent.id)[:8] if intent else 'open-loop'}"
        ),
    )
    db.add(ledger_entry)

    await db.commit()

    available_after = float(balance_after) - float(reserved_after)
    log.info(
        "micropay_settled",
        tx_id=str(micro_tx.id),
        idempotency_key=request.idempotency_key,
        amount=str(request.amount_usdc),
        loop_type=str(wallet.loop_type),
        payee=request.payee_address,
    )

    return MicroPayResponse(
        transaction_id=micro_tx.id,
        idempotency_key=micro_tx.idempotency_key,
        status=micro_tx.status,
        amount_usdc=micro_tx.amount_usdc,
        loop_type=str(micro_tx.loop_type),
        payment_intent_id=micro_tx.payment_intent_id,
        payer_available_balance_after=Decimal(str(max(available_after, 0))),
        settled_at=settled_at.isoformat(),
        on_chain_tx_hash=None,
    )
