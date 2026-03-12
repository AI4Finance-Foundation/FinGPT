"""
Agent Ledger models — stablecoin micro-transaction layer.

Actor hierarchy:
  AgentOwner (human, FK → clients.id)
    └── AgentAccount (master agent — no KYB; must complete KYA)
          └── AgentWallet (sub-wallet; loop_type = open | closed; spending rules)
                └── PaymentIntent (closed-loop only; records purpose before payment)

Settlement flow:
  Fiat → [v1 Ingress + Bridge] → USDC top-up → AgentAccount.prefund_balance_usdc
  AgentAccount → [micropay] → MicroTransaction → off-chain settle → on-chain sweep (Base)
  AgentAccount → [withdrawal] → [v1 Routing Engine + corridor adapter] → fiat to owner

Zero modifications to any v1 model.
"""
import uuid
from enum import Enum as PyEnum
from sqlalchemy import String, Boolean, Numeric, Integer, Text, ForeignKey, Enum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin, UUIDMixin


# ── Enums ──────────────────────────────────────────────────────────────────────

class KYAStatus(str, PyEnum):
    UNVERIFIED = "unverified"   # Cannot transact
    PENDING    = "pending"      # Submitted; awaiting validator stamp; top-up only
    BASIC      = "basic"        # Owner identity + purpose confirmed; ≤ $1/day
    ENHANCED   = "enhanced"     # Extended due diligence; ≤ $100/day


class AgentStatus(str, PyEnum):
    ACTIVE      = "active"
    SUSPENDED   = "suspended"
    TERMINATED  = "terminated"


class LoopType(str, PyEnum):
    OPEN   = "open"    # No intent required; free within spending rules
    CLOSED = "closed"  # Must have a PaymentIntent before every payment


class IntentStatus(str, PyEnum):
    RESERVED   = "reserved"    # Amount held; awaiting consumption
    CONSUMED   = "consumed"    # Payment made; linked to micro_transaction
    EXPIRED    = "expired"     # Unused past expires_at; refund triggered
    REFUNDED   = "refunded"    # Reserved amount returned to owner
    SUPERSEDED = "superseded"  # Business purpose changed; new intent created


class IntentCategory(str, PyEnum):
    COMPUTE       = "compute"
    DATA          = "data"
    AGENT_SERVICE = "agent_service"
    CONTENT       = "content"
    OFFRAMP       = "offramp"   # Agent requesting fiat withdrawal via v1 rails
    OTHER         = "other"


class MicroTxStatus(str, PyEnum):
    PENDING          = "pending"
    SETTLED_OFFCHAIN = "settled_offchain"  # DB settled; awaiting on-chain sweep
    SETTLED_ONCHAIN  = "settled_onchain"   # On-chain USDC transfer confirmed
    FAILED           = "failed"
    REFUNDED         = "refunded"


# ── Models ─────────────────────────────────────────────────────────────────────

class AgentAccount(Base, UUIDMixin, TimestampMixin):
    """
    Master agent entity.

    No KYB required — agents are software. But KYA (Know Your Agent) is
    mandatory before any outbound payment. The owner_client_id links this
    agent to the human AgentOwner (an existing v1 Client).

    prefund_balance_usdc  — spendable balance (top-ups in, payments out)
    reserved_balance_usdc — held by active closed-loop PaymentIntents;
                            cannot be spent until the intent is consumed or released
    """
    __tablename__ = "agent_accounts"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    owner_client_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clients.id", ondelete="SET NULL")
    )
    api_key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    status: Mapped[AgentStatus] = mapped_column(
        Enum(AgentStatus), default=AgentStatus.ACTIVE, nullable=False
    )
    # Denormalized from agent_kya for fast compliance gate checks
    kya_status: Mapped[KYAStatus] = mapped_column(
        Enum(KYAStatus), default=KYAStatus.UNVERIFIED, nullable=False
    )
    chain: Mapped[str] = mapped_column(String(32), default="base", nullable=False)
    prefund_balance_usdc: Mapped[float] = mapped_column(
        Numeric(28, 8), default=0, nullable=False
    )
    reserved_balance_usdc: Mapped[float] = mapped_column(
        Numeric(28, 8), default=0, nullable=False
    )
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    kya: Mapped["AgentKYA | None"] = relationship(
        back_populates="agent_account", uselist=False
    )
    wallets: Mapped[list["AgentWallet"]] = relationship(
        back_populates="agent_account", cascade="all, delete-orphan"
    )
    ledger_entries: Mapped[list["AgentLedgerEntry"]] = relationship(
        back_populates="agent_account"
    )


class AgentKYA(Base, UUIDMixin, TimestampMixin):
    """
    Know Your Agent record.

    Every AgentAccount must complete KYA before transacting. A third-party
    validator issues a stamped token (JWT or opaque string) confirming:
      - Owner identity and control of the agent
      - Agent's declared purpose and use case
      - No sanctions matches

    KYA levels gate transaction limits:
      basic    → ≤ $1/day aggregate
      enhanced → ≤ $100/day aggregate

    validator_token is stored encrypted (GCP Secret Manager in production;
    stored as opaque string here with encryption handled by the service layer).
    """
    __tablename__ = "agent_kya"

    agent_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_accounts.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    status: Mapped[KYAStatus] = mapped_column(
        Enum(KYAStatus), default=KYAStatus.UNVERIFIED, nullable=False
    )
    kya_level: Mapped[KYAStatus] = mapped_column(
        Enum(KYAStatus), default=KYAStatus.UNVERIFIED, nullable=False
    )

    # Third-party validator
    validator_name: Mapped[str | None] = mapped_column(String(128))
    validator_ref: Mapped[str | None] = mapped_column(String(255))
    validator_token: Mapped[str | None] = mapped_column(Text)  # encrypted in prod
    validator_expires_at: Mapped[str | None] = mapped_column(String(64))
    validated_at: Mapped[str | None] = mapped_column(String(64))
    last_reviewed_at: Mapped[str | None] = mapped_column(String(64))

    # Agent identity fields (captured at KYA submission)
    agent_purpose: Mapped[str | None] = mapped_column(Text)
    agent_owner_attestation: Mapped[str | None] = mapped_column(Text)
    declared_use_case: Mapped[str | None] = mapped_column(String(64))
    # data_retrieval | content_generation | trading_support | general

    submitted_at: Mapped[str | None] = mapped_column(String(64))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    agent_account: Mapped["AgentAccount"] = relationship(back_populates="kya")


class AgentWallet(Base, UUIDMixin, TimestampMixin):
    """
    Sub-wallet under a MasterAgent (AgentAccount).

    loop_type is set by the AgentOwner at creation and is IMMUTABLE.
    - open:   payments allowed without a PaymentIntent
    - closed: every payment must reference a valid PaymentIntent; unused
              intents are automatically refunded to the AgentOwner

    Spending rules are enforced off-chain at micropay time.
    """
    __tablename__ = "agent_wallets"

    agent_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_accounts.id", ondelete="CASCADE"),
        nullable=False,
    )
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    wallet_address: Mapped[str] = mapped_column(String(255), nullable=False)
    chain: Mapped[str] = mapped_column(String(32), default="base", nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)
    loop_type: Mapped[LoopType] = mapped_column(
        Enum(LoopType), default=LoopType.OPEN, nullable=False
    )

    # Spending rules
    max_per_tx_usdc: Mapped[float] = mapped_column(Numeric(28, 8), default=0.10)
    max_daily_usdc: Mapped[float] = mapped_column(Numeric(28, 8), default=1.00)
    daily_spent_usdc: Mapped[float] = mapped_column(Numeric(28, 8), default=0.0)
    daily_reset_at: Mapped[str | None] = mapped_column(String(64))
    allowed_counterparties: Mapped[list] = mapped_column(JSONB, default=list)
    expires_at: Mapped[str | None] = mapped_column(String(64))
    max_uses: Mapped[int | None] = mapped_column(Integer)
    use_count: Mapped[int] = mapped_column(Integer, default=0)

    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    agent_account: Mapped["AgentAccount"] = relationship(back_populates="wallets")
    payment_intents: Mapped[list["PaymentIntent"]] = relationship(
        back_populates="payer_wallet", cascade="all, delete-orphan"
    )
    outbound_transactions: Mapped[list["MicroTransaction"]] = relationship(
        foreign_keys="MicroTransaction.payer_wallet_id",
        back_populates="payer_wallet",
    )


class PaymentIntent(Base, UUIDMixin, TimestampMixin):
    """
    Closed-loop payment intent.

    Created BEFORE the payment to record the declared purpose of the
    transaction. Mandatory when AgentWallet.loop_type == 'closed'.

    Lifecycle:
      reserved → consumed (payment made)
               → expired  (unused past expires_at; refund triggered by intent_sweeper)
               → refunded (reserved amount returned to AgentOwner)
               → superseded (business purpose changed; audit trail preserved)

    The 'superseded_by_intent_id' self-FK links the chain of intent revisions.
    No intent record is ever deleted.
    """
    __tablename__ = "payment_intents"

    payer_wallet_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_wallets.id", ondelete="CASCADE"),
        nullable=False,
    )
    amount_usdc: Mapped[float] = mapped_column(Numeric(28, 8), nullable=False)
    asset: Mapped[str] = mapped_column(String(16), default="USDC", nullable=False)
    intent_description: Mapped[str] = mapped_column(Text, nullable=False)
    intent_category: Mapped[IntentCategory] = mapped_column(
        Enum(IntentCategory), default=IntentCategory.OTHER, nullable=False
    )
    status: Mapped[IntentStatus] = mapped_column(
        Enum(IntentStatus), default=IntentStatus.RESERVED, nullable=False
    )
    expires_at: Mapped[str] = mapped_column(String(64), nullable=False)

    # Populated on consumption
    consumed_micro_tx_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("micro_transactions.id")
    )
    # Populated if superseded (points to replacement intent)
    superseded_by_intent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("payment_intents.id")
    )
    # Populated if refunded
    refund_tx_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("micro_transactions.id")
    )
    audit_note: Mapped[str | None] = mapped_column(Text)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    payer_wallet: Mapped["AgentWallet"] = relationship(back_populates="payment_intents")


class MicroTransaction(Base, UUIDMixin, TimestampMixin):
    """
    Single stablecoin micro-payment between two addresses.

    NOT a batch. Settled synchronously (off-chain in DB) and then swept
    on-chain periodically by chain_watcher.py.

    idempotency_key is caller-supplied and enforced as a DB UNIQUE constraint
    to prevent double-spend on retries.

    loop_type mirrors the payer wallet's loop_type at time of settlement
    for reporting purposes (immutable once created).
    """
    __tablename__ = "micro_transactions"

    idempotency_key: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    payer_wallet_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_wallets.id"),
        nullable=False,
    )
    payee_address: Mapped[str] = mapped_column(String(255), nullable=False)
    payee_wallet_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("agent_wallets.id")
    )
    amount_usdc: Mapped[float] = mapped_column(Numeric(28, 8), nullable=False)
    chain: Mapped[str] = mapped_column(String(32), default="base", nullable=False)
    loop_type: Mapped[LoopType] = mapped_column(
        Enum(LoopType), nullable=False
    )
    payment_intent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("payment_intents.id")
    )
    # x402 linkage
    x402_payment_header: Mapped[str | None] = mapped_column(Text)
    x402_resource_url: Mapped[str | None] = mapped_column(String(2048))

    status: Mapped[MicroTxStatus] = mapped_column(
        Enum(MicroTxStatus), default=MicroTxStatus.PENDING, nullable=False
    )
    on_chain_tx_hash: Mapped[str | None] = mapped_column(String(255))
    on_chain_block: Mapped[int | None] = mapped_column(Integer)
    on_chain_confirmed_at: Mapped[str | None] = mapped_column(String(64))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    settled_at: Mapped[str | None] = mapped_column(String(64))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    payer_wallet: Mapped["AgentWallet"] = relationship(
        foreign_keys=[payer_wallet_id], back_populates="outbound_transactions"
    )


class AgentLedgerEntry(Base, UUIDMixin, TimestampMixin):
    """
    Append-only double-entry ledger for AgentAccount balances.

    entry_type values:
      credit          — top-up deposit confirmed on-chain
      debit           — micro-payment settled
      refund          — expired intent amount returned
      fee             — Finogrid service fee (future)
      intent_reserve  — amount moved to reserved_balance on intent creation
      intent_release  — amount released from reserved_balance (consumed or expired)

    balance_after and reserved_balance_after are snapshots taken at write time
    for audit purposes. Do not use them for live balance queries; always read
    from agent_accounts.prefund_balance_usdc and reserved_balance_usdc.
    """
    __tablename__ = "agent_ledger_entries"

    agent_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_accounts.id"),
        nullable=False,
    )
    entry_type: Mapped[str] = mapped_column(String(32), nullable=False)
    amount_usdc: Mapped[float] = mapped_column(Numeric(28, 8), nullable=False)
    balance_after: Mapped[float] = mapped_column(Numeric(28, 8), nullable=False)
    reserved_balance_after: Mapped[float] = mapped_column(Numeric(28, 8), nullable=False)

    micro_tx_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("micro_transactions.id")
    )
    payment_intent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("payment_intents.id")
    )
    on_chain_tx_hash: Mapped[str | None] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)

    agent_account: Mapped["AgentAccount"] = relationship(back_populates="ledger_entries")
