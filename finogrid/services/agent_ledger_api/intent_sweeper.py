"""
Intent Sweeper — background job for closed-loop intent lifecycle management.

Runs every settings.intent_sweeper_interval_seconds (default: 300).
Finds all PaymentIntents with status=reserved and expires_at < now, then:
  1. Transitions status → expired
  2. Releases reserved_balance_usdc on the AgentAccount
  3. Creates a refund AgentLedgerEntry (entry_type=intent_release)
  4. Appends to audit_logs

This ensures the closed-loop guarantee: unused reserved funds ALWAYS return
to the AgentOwner's prefund balance. No silent forfeitures.
"""
import asyncio
import structlog
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

from .config import settings
from ....database.models.agent_ledger import (
    AgentAccount, AgentWallet, PaymentIntent, AgentLedgerEntry, IntentStatus
)
from ....database.models.audit import AuditLog

log = structlog.get_logger()

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def sweep_expired_intents(db: AsyncSession) -> int:
    """
    Find and process all expired reserved intents.
    Returns count of intents processed.
    """
    now = datetime.now(timezone.utc)

    result = await db.execute(
        select(PaymentIntent).where(
            PaymentIntent.status == IntentStatus.RESERVED,
            PaymentIntent.expires_at < now,
        )
    )
    expired_intents = result.scalars().all()

    processed = 0
    for intent in expired_intents:
        try:
            # Load the wallet to get agent_account_id
            wallet_result = await db.execute(
                select(AgentWallet).where(AgentWallet.id == intent.payer_wallet_id)
            )
            wallet = wallet_result.scalar_one_or_none()
            if wallet is None:
                log.warning("intent_sweeper_wallet_missing", intent_id=str(intent.id))
                continue

            # Load agent account
            agent_result = await db.execute(
                select(AgentAccount).where(AgentAccount.id == wallet.agent_account_id)
            )
            agent = agent_result.scalar_one_or_none()
            if agent is None:
                log.warning("intent_sweeper_agent_missing", intent_id=str(intent.id))
                continue

            # Transition intent to expired
            intent.status = IntentStatus.EXPIRED
            intent.audit_note = (
                intent.audit_note or ""
            ) + f" | Auto-expired at {now.isoformat()} by intent_sweeper"

            # Release reserved balance
            release_amount = intent.amount_usdc
            agent.reserved_balance_usdc = max(
                Decimal("0"),
                Decimal(str(agent.reserved_balance_usdc)) - Decimal(str(release_amount)),
            )

            balance_after = Decimal(str(agent.prefund_balance_usdc))
            reserved_after = Decimal(str(agent.reserved_balance_usdc))

            # Ledger entry: intent_release
            ledger_entry = AgentLedgerEntry(
                agent_account_id=agent.id,
                entry_type="intent_release",
                amount_usdc=release_amount,
                balance_after=balance_after,
                reserved_balance_after=reserved_after,
                payment_intent_id=intent.id,
                description=(
                    f"Intent {str(intent.id)[:8]}... expired. "
                    f"{float(release_amount):.4f} USDC released from reservation."
                ),
            )
            db.add(ledger_entry)

            # Audit log
            audit_entry = AuditLog(
                entity_type="payment_intent",
                entity_id=str(intent.id),
                action="expired",
                actor="intent_sweeper",
                details={
                    "agent_account_id": str(agent.id),
                    "wallet_id": str(wallet.id),
                    "amount_usdc": str(release_amount),
                    "expired_at": now.isoformat(),
                    "intent_description": intent.intent_description,
                    "intent_category": intent.intent_category,
                },
            )
            db.add(audit_entry)

            log.info(
                "intent_expired_and_released",
                intent_id=str(intent.id),
                agent_id=str(agent.id),
                amount=str(release_amount),
            )
            processed += 1

        except Exception as exc:  # noqa: BLE001
            log.error("intent_sweeper_error", intent_id=str(intent.id), error=str(exc))
            continue

    if processed > 0:
        await db.commit()

    return processed


async def run_sweeper_loop():
    """Main sweeper loop — runs indefinitely."""
    log.info("intent_sweeper_started", interval=settings.intent_sweeper_interval_seconds)
    while True:
        try:
            async with AsyncSessionLocal() as db:
                count = await sweep_expired_intents(db)
                if count > 0:
                    log.info("intent_sweep_completed", expired_count=count)
        except Exception as exc:  # noqa: BLE001
            log.error("intent_sweeper_loop_error", error=str(exc))

        await asyncio.sleep(settings.intent_sweeper_interval_seconds)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_sweeper_loop())
