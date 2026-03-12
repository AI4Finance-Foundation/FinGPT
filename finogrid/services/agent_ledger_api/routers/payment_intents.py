"""Payment intent endpoints — closed-loop only."""
import structlog
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..schemas import PaymentIntentCreateRequest, PaymentIntentCreateResponse, PaymentIntentSupersede
from ..dependencies import get_db, get_current_agent_account
from .....database.models.agent_ledger import (
    AgentAccount, AgentWallet, PaymentIntent, IntentStatus, LoopType
)

log = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=PaymentIntentCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_payment_intent(
    request: PaymentIntentCreateRequest,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """
    Create a PaymentIntent, reserving funds for a future closed-loop payment.
    Only closed-loop wallets require this; open-loop wallets reject this endpoint.
    The reserved amount is locked in agent.reserved_balance_usdc until consumed or expired.
    """
    # Load and validate wallet ownership
    wallet_result = await db.execute(
        select(AgentWallet).where(
            AgentWallet.id == request.payer_wallet_id,
            AgentWallet.agent_account_id == agent.id,
        )
    )
    wallet = wallet_result.scalar_one_or_none()
    if wallet is None:
        raise HTTPException(status_code=404, detail="Wallet not found or access denied")

    if wallet.loop_type != LoopType.CLOSED:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Wallet {wallet.id} is open-loop. "
                "Payment intents are only required for closed-loop wallets. "
                "Use POST /v1/micropay directly for open-loop payments."
            ),
        )

    if wallet.status != "active":
        raise HTTPException(status_code=400, detail=f"Wallet status is '{wallet.status}', not active")

    # Validate expiry
    now = datetime.now(timezone.utc)
    if request.expires_at <= now:
        raise HTTPException(status_code=400, detail="expires_at must be in the future")

    # Check available balance covers the reservation
    available = float(agent.prefund_balance_usdc) - float(agent.reserved_balance_usdc)
    if available < float(request.amount_usdc):
        raise HTTPException(
            status_code=402,
            detail=(
                f"Insufficient available balance. "
                f"Available: {available:.2f} USDC, "
                f"Requested: {float(request.amount_usdc):.2f} USDC. "
                "Top up via POST /v1/agent-accounts/{id}/topup."
            ),
        )

    # Create intent and reserve funds atomically
    intent = PaymentIntent(
        payer_wallet_id=request.payer_wallet_id,
        amount_usdc=request.amount_usdc,
        intent_description=request.intent_description,
        intent_category=request.intent_category,
        status=IntentStatus.RESERVED,
        expires_at=request.expires_at,
    )
    db.add(intent)

    from decimal import Decimal
    agent.reserved_balance_usdc = Decimal(str(agent.reserved_balance_usdc)) + request.amount_usdc

    await db.commit()
    await db.refresh(intent)

    log.info(
        "payment_intent_created",
        intent_id=str(intent.id),
        wallet_id=str(wallet.id),
        amount=str(request.amount_usdc),
        category=request.intent_category,
    )
    return PaymentIntentCreateResponse(
        payment_intent_id=intent.id,
        payer_wallet_id=intent.payer_wallet_id,
        amount_usdc=intent.amount_usdc,
        intent_category=intent.intent_category,
        expires_at=intent.expires_at.isoformat(),
    )


@router.patch("/{intent_id}", response_model=PaymentIntentCreateResponse)
async def supersede_payment_intent(
    intent_id: str,
    request: PaymentIntentSupersede,
    db: AsyncSession = Depends(get_db),
    agent: AgentAccount = Depends(get_current_agent_account),
):
    """
    Supersede an existing reserved intent — used when purpose or amount changes.
    The old intent is preserved with status=superseded and a pointer to the new one.
    An audit_note is REQUIRED to explain why the intent changed.
    Closed-loop compliance: intent chain is always reconstructable.
    """
    intent_result = await db.execute(
        select(PaymentIntent).where(PaymentIntent.id == intent_id)
    )
    old_intent = intent_result.scalar_one_or_none()
    if old_intent is None:
        raise HTTPException(status_code=404, detail="PaymentIntent not found")

    # Verify the wallet belongs to this agent
    wallet_result = await db.execute(
        select(AgentWallet).where(
            AgentWallet.id == old_intent.payer_wallet_id,
            AgentWallet.agent_account_id == agent.id,
        )
    )
    if wallet_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=403, detail="Access denied")

    if old_intent.status != IntentStatus.RESERVED:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot supersede intent with status '{old_intent.status}'. Only 'reserved' intents can be superseded.",
        )

    now = datetime.now(timezone.utc)
    if request.new_expires_at <= now:
        raise HTTPException(status_code=400, detail="new_expires_at must be in the future")

    # Adjust reserved balance for amount difference
    from decimal import Decimal
    amount_delta = request.new_amount_usdc - old_intent.amount_usdc
    available = float(agent.prefund_balance_usdc) - float(agent.reserved_balance_usdc)
    if float(amount_delta) > available:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Insufficient balance for increased reservation. "
                f"Delta: {float(amount_delta):.2f} USDC, Available: {available:.2f} USDC."
            ),
        )

    # Create new intent
    new_intent = PaymentIntent(
        payer_wallet_id=old_intent.payer_wallet_id,
        amount_usdc=request.new_amount_usdc,
        intent_description=request.new_intent_description,
        intent_category=request.new_intent_category,
        status=IntentStatus.RESERVED,
        expires_at=request.new_expires_at,
        audit_note=request.audit_note,
    )
    db.add(new_intent)
    await db.flush()  # Get new_intent.id before referencing

    # Mark old intent as superseded
    old_intent.status = IntentStatus.SUPERSEDED
    old_intent.superseded_by_intent_id = new_intent.id
    old_intent.audit_note = request.audit_note

    # Update reserved balance
    agent.reserved_balance_usdc = Decimal(str(agent.reserved_balance_usdc)) + amount_delta

    await db.commit()
    await db.refresh(new_intent)

    log.info(
        "payment_intent_superseded",
        old_intent_id=intent_id,
        new_intent_id=str(new_intent.id),
        audit_note=request.audit_note,
    )
    return PaymentIntentCreateResponse(
        payment_intent_id=new_intent.id,
        payer_wallet_id=new_intent.payer_wallet_id,
        amount_usdc=new_intent.amount_usdc,
        intent_category=new_intent.intent_category,
        expires_at=new_intent.expires_at.isoformat(),
        message=f"Intent superseded. Old intent ID: {intent_id} preserved with status=superseded.",
    )
