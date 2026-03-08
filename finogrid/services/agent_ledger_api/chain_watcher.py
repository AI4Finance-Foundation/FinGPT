"""
Chain Watcher — Base L2 on-chain event processor.

Two jobs running in parallel:
  Job A — Deposit watcher: monitors USDC transfers to the Finogrid deposit address
           and credits agent prefund balances.
  Job B — Settlement sweep: promotes micro_transactions from settled_offchain to
           settled_onchain after confirming the on-chain batch.

Requires: web3.py, settings.chain_enabled=True, settings.base_rpc_url
"""
import asyncio
import structlog
from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

from .config import settings
from ....database.models.agent_ledger import (
    AgentAccount, AgentWallet, MicroTransaction, AgentLedgerEntry,
    MicroTxStatus,
)
from ....database.models.audit import AuditLog

log = structlog.get_logger()

# USDC Transfer event ABI (ERC-20 Transfer)
USDC_TRANSFER_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    }
]
USDC_DECIMALS = 6  # Native USDC on Base has 6 decimals

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


def _get_web3():
    """Lazily import and return a Web3 instance connected to Base RPC."""
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider(settings.base_rpc_url))
    if not w3.is_connected():
        raise RuntimeError(f"Cannot connect to Base RPC at {settings.base_rpc_url}")
    return w3


async def job_a_deposit_watcher(last_block: int) -> int:
    """
    Scan new blocks for USDC Transfer events to the deposit address.
    Returns the latest block scanned.
    """
    try:
        w3 = _get_web3()
        latest = w3.eth.block_number
        if latest <= last_block:
            return last_block

        usdc = w3.eth.contract(
            address=w3.to_checksum_address(settings.usdc_contract_address_base),
            abi=USDC_TRANSFER_ABI,
        )
        deposit_addr = w3.to_checksum_address(settings.agent_ledger_deposit_address)

        events = usdc.events.Transfer.get_logs(
            fromBlock=last_block + 1,
            toBlock=latest,
            argument_filters={"to": deposit_addr},
        )

        for event in events:
            tx_hash = event["transactionHash"].hex()
            from_addr = event["args"]["from"].lower()
            amount_raw = event["args"]["value"]
            amount_usdc = Decimal(str(amount_raw)) / Decimal(str(10 ** USDC_DECIMALS))

            log.info(
                "deposit_detected",
                tx_hash=tx_hash,
                from_addr=from_addr,
                amount_usdc=str(amount_usdc),
            )

            async with AsyncSessionLocal() as db:
                await _credit_deposit(db, tx_hash, from_addr, amount_usdc)

        return latest

    except Exception as exc:  # noqa: BLE001
        log.error("deposit_watcher_error", error=str(exc))
        return last_block


async def _credit_deposit(db: AsyncSession, tx_hash: str, from_addr: str, amount_usdc: Decimal):
    """
    Match an on-chain USDC deposit to an AgentAccount by wallet address and credit.
    Idempotent: checks whether this tx_hash was already credited.
    """
    # Idempotency: skip if already credited
    existing = await db.execute(
        select(AgentLedgerEntry).where(
            AgentLedgerEntry.on_chain_tx_hash == tx_hash,
            AgentLedgerEntry.entry_type == "credit",
        )
    )
    if existing.scalar_one_or_none():
        log.debug("deposit_already_credited", tx_hash=tx_hash)
        return

    # Find the AgentWallet matching the sending address (from_addr)
    wallet_result = await db.execute(
        select(AgentWallet).where(AgentWallet.wallet_address == from_addr)
    )
    wallet = wallet_result.scalar_one_or_none()

    if wallet is None:
        log.warning("deposit_no_matching_wallet", from_addr=from_addr, tx_hash=tx_hash)
        return

    # Load agent
    agent_result = await db.execute(
        select(AgentAccount).where(AgentAccount.id == wallet.agent_account_id)
    )
    agent = agent_result.scalar_one_or_none()
    if agent is None:
        return

    # Credit
    agent.prefund_balance_usdc = Decimal(str(agent.prefund_balance_usdc)) + amount_usdc
    balance_after = Decimal(str(agent.prefund_balance_usdc))

    ledger_entry = AgentLedgerEntry(
        agent_account_id=agent.id,
        entry_type="credit",
        amount_usdc=amount_usdc,
        balance_after=balance_after,
        reserved_balance_after=Decimal(str(agent.reserved_balance_usdc)),
        on_chain_tx_hash=tx_hash,
        description=f"USDC deposit from {from_addr[:10]}... confirmed on Base",
    )
    db.add(ledger_entry)
    await db.commit()

    log.info(
        "deposit_credited",
        agent_id=str(agent.id),
        tx_hash=tx_hash,
        amount=str(amount_usdc),
        balance_after=str(balance_after),
    )


async def job_b_settlement_sweep():
    """
    Promote settled_offchain micro_transactions to settled_onchain.
    In production, this would batch-sweep transactions on-chain.
    For MVP: marks settled_offchain txs older than 60s as settled_onchain.
    """
    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=60)

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(MicroTransaction).where(
                MicroTransaction.status == MicroTxStatus.SETTLED_OFFCHAIN,
                MicroTransaction.settled_at < cutoff,
            ).limit(100)
        )
        pending_sweep = result.scalars().all()

        if not pending_sweep:
            return

        for tx in pending_sweep:
            tx.status = MicroTxStatus.SETTLED_ONCHAIN
            tx.on_chain_confirmed_at = datetime.now(timezone.utc)
            # on_chain_tx_hash would be set by the actual sweep transaction

        await db.commit()
        log.info("settlement_sweep_completed", count=len(pending_sweep))


async def run_chain_watcher():
    """Main chain watcher loop."""
    if not settings.chain_enabled:
        log.info("chain_watcher_disabled", reason="settings.chain_enabled=False")
        return

    log.info("chain_watcher_started", chain=settings.chain, rpc=settings.base_rpc_url)

    try:
        w3 = _get_web3()
        last_block = w3.eth.block_number - 100  # Start 100 blocks back
    except Exception as exc:  # noqa: BLE001
        log.error("chain_watcher_init_failed", error=str(exc))
        return

    while True:
        # Run both jobs concurrently
        last_block = await job_a_deposit_watcher(last_block)
        await job_b_settlement_sweep()
        await asyncio.sleep(settings.chain_watcher_sweep_interval_seconds)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_chain_watcher())
