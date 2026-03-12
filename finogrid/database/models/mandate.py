"""
Mandate models — delegation and authority layer.

The Mandate is the core authorisation primitive in the Finogrid platform.
Every agent action must be traceable to a valid, active Mandate granted by a Principal.

Actor hierarchy:
  Principal  (human or organisation — maps to Client in v1, or standalone)
    └── AgentAccount  (software entity; must have Mandate to act)
          └── Mandate  (scoped delegation: what the agent can do, for how long, up to how much)
                └── MandateEvent  (append-only audit trail of every mandate lifecycle change)

Design rules:
  1. A Mandate is IMMUTABLE once activated — changes require supersession (new mandate + old archived)
  2. Every money movement must reference a valid Mandate at execution time
  3. Revocation is immediate and propagates to all in-flight transactions
  4. Mandate scope defines: corridors, assets, counterparties, amount caps, approval thresholds
"""
import uuid
from enum import Enum as PyEnum
from typing import Optional
from sqlalchemy import String, Numeric, Integer, Text, ForeignKey, Enum, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import TIMESTAMP
from .base import Base, TimestampMixin, UUIDMixin


# ── Enums ─────────────────────────────────────────────────────────────────────

class MandateStatus(str, PyEnum):
    DRAFT      = "draft"       # Created but not yet activated
    ACTIVE     = "active"      # Grants authority to agent
    SUSPENDED  = "suspended"   # Temporarily blocked; can be reactivated
    REVOKED    = "revoked"     # Permanently cancelled; cannot be reactivated
    EXPIRED    = "expired"     # Past expires_at; auto-transitioned by sweeper
    SUPERSEDED = "superseded"  # Replaced by a newer mandate; old record preserved


class MandateScope(str, PyEnum):
    PAYOUT    = "payout"     # Agent may initiate outbound payments
    COLLECT   = "collect"    # Agent may initiate collections
    TOPUP     = "topup"      # Agent may top up prefund balance
    FULL      = "full"       # All of the above (max privilege; requires enhanced KYA)
    READ_ONLY = "read_only"  # Agent may query balances and status only


class ApprovalMode(str, PyEnum):
    AUTO      = "auto"       # All transactions within limits auto-approved
    MANUAL    = "manual"     # All transactions require human approval
    THRESHOLD = "threshold"  # Auto below threshold; manual above


class MandateEventType(str, PyEnum):
    CREATED    = "created"
    ACTIVATED  = "activated"
    SUSPENDED  = "suspended"
    RESUMED    = "resumed"
    REVOKED    = "revoked"
    EXPIRED    = "expired"
    SUPERSEDED = "superseded"
    LIMIT_HIT  = "limit_hit"   # Audit: a transaction was blocked by this mandate's limits


# ── Models ────────────────────────────────────────────────────────────────────

class Principal(Base, UUIDMixin, TimestampMixin):
    """
    The authorising human or entity behind an agent.
    In v1: AgentOwner (human) maps to Client. This model exists to support
    standalone principals not yet onboarded as v1 Clients (e.g., FI counterparties
    or agent-network members).
    """
    __tablename__ = "principals"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(255))
    client_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clients.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Link to v1 Client record if onboarded"
    )
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="active",
        comment="active | suspended | terminated"
    )
    kyb_status: Mapped[str] = mapped_column(
        String(50), nullable=False, default="pending",
        comment="Mirrors clients.kyb_status; denormalized for fast gate checks"
    )
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    mandates: Mapped[list["Mandate"]] = relationship("Mandate", back_populates="principal")


class Mandate(Base, UUIDMixin, TimestampMixin):
    """
    Authorisation grant from a Principal to an AgentAccount.

    A Mandate defines:
      - what the agent can do (scope)
      - within what limits (amount caps, corridors, assets, counterparties)
      - for how long (expires_at)
      - under what approval mode (auto / manual / threshold)
      - with what approval threshold (above this amount → human approval required)

    Immutability: once activated, PATCH creates a new mandate + supersedes the old.
    """
    __tablename__ = "mandates"

    principal_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("principals.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    agent_account_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_accounts.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        Enum(MandateStatus, name="mandate_status"),
        nullable=False,
        default=MandateStatus.DRAFT,
    )
    scope: Mapped[str] = mapped_column(
        Enum(MandateScope, name="mandate_scope"),
        nullable=False,
        default=MandateScope.PAYOUT,
        comment="What the agent is authorised to do under this mandate",
    )
    approval_mode: Mapped[str] = mapped_column(
        Enum(ApprovalMode, name="approval_mode"),
        nullable=False,
        default=ApprovalMode.AUTO,
    )

    # ── Amount limits ─────────────────────────────────────────────────────────
    max_amount_per_tx_usdc: Mapped[Optional[float]] = mapped_column(
        Numeric(28, 8),
        nullable=True,
        comment="Hard cap per single transaction. NULL = no limit (requires FULL scope)"
    )
    max_daily_usdc: Mapped[Optional[float]] = mapped_column(
        Numeric(28, 8),
        nullable=True,
        comment="Aggregate daily cap across all wallets under this mandate"
    )
    max_monthly_usdc: Mapped[Optional[float]] = mapped_column(
        Numeric(28, 8),
        nullable=True,
    )
    approval_threshold_usdc: Mapped[Optional[float]] = mapped_column(
        Numeric(28, 8),
        nullable=True,
        comment="Transactions at or above this amount require human approval (THRESHOLD mode)"
    )
    lifetime_cap_usdc: Mapped[Optional[float]] = mapped_column(
        Numeric(28, 8),
        nullable=True,
        comment="Total spend ceiling for the lifetime of this mandate"
    )
    lifetime_spent_usdc: Mapped[float] = mapped_column(
        Numeric(28, 8), nullable=False, default=0,
    )

    # ── Scope constraints ─────────────────────────────────────────────────────
    allowed_corridors: Mapped[Optional[list]] = mapped_column(
        JSONB, nullable=True,
        comment='["BR", "NG", "IN"] — empty/null means all corridors permitted'
    )
    allowed_assets: Mapped[Optional[list]] = mapped_column(
        JSONB, nullable=True,
        comment='["USDC", "USDT"] — empty/null means all assets'
    )
    allowed_chains: Mapped[Optional[list]] = mapped_column(
        JSONB, nullable=True,
        comment='["base", "polygon"] — empty/null means all chains'
    )
    allowed_counterparties: Mapped[Optional[list]] = mapped_column(
        JSONB, nullable=True,
        comment="Strict allowlist of payee addresses or wallet IDs. Empty = any."
    )

    # ── Validity ──────────────────────────────────────────────────────────────
    activated_at: Mapped[Optional[object]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    expires_at: Mapped[Optional[object]] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True,
        comment="NULL = no expiry. Set for time-bounded agent authority."
    )
    revoked_at: Mapped[Optional[object]] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    revocation_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Supersession chain ────────────────────────────────────────────────────
    superseded_by_mandate_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mandates.id", ondelete="SET NULL"),
        nullable=True,
        comment="Points to the replacement mandate when this one is superseded"
    )
    supersession_note: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="Required explanation of why the mandate was changed"
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    principal: Mapped["Principal"] = relationship("Principal", back_populates="mandates")
    events: Mapped[list["MandateEvent"]] = relationship(
        "MandateEvent", back_populates="mandate", order_by="MandateEvent.created_at"
    )


class MandateEvent(Base, UUIDMixin, TimestampMixin):
    """
    Append-only audit trail for every mandate lifecycle change.
    Never updated, only inserted.
    """
    __tablename__ = "mandate_events"

    mandate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("mandates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(
        Enum(MandateEventType, name="mandate_event_type"),
        nullable=False,
    )
    actor: Mapped[str] = mapped_column(
        String(255), nullable=False,
        comment="Who triggered the event: principal_id, agent_account_id, or system service name"
    )
    previous_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    new_status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    mandate: Mapped["Mandate"] = relationship("Mandate", back_populates="events")
