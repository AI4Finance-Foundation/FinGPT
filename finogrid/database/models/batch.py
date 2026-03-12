"""Payout batch and task models."""
import uuid
from enum import Enum as PyEnum
from sqlalchemy import String, Numeric, Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin, UUIDMixin


class BatchStatus(str, PyEnum):
    DRAFT = "draft"
    VALIDATING = "validating"
    PENDING_COMPLIANCE = "pending_compliance"
    PROCESSING = "processing"
    PARTIALLY_COMPLETED = "partially_completed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, PyEnum):
    PENDING = "pending"
    ROUTING = "routing"
    COMPLIANCE_CHECK = "compliance_check"
    HELD_FOR_REVIEW = "held_for_review"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class AssetType(str, PyEnum):
    USDT = "USDT"
    USDC = "USDC"


class DeliveryMode(str, PyEnum):
    WALLET = "wallet"
    FIAT = "fiat"


class Batch(Base, UUIDMixin, TimestampMixin):
    """A payout batch submitted by a business client."""

    __tablename__ = "batches"

    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clients.id"), nullable=False
    )
    reference: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[BatchStatus] = mapped_column(
        Enum(BatchStatus), default=BatchStatus.DRAFT, nullable=False
    )
    total_amount_usd: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    task_count: Mapped[int] = mapped_column(default=0)
    completed_count: Mapped[int] = mapped_column(default=0)
    failed_count: Mapped[int] = mapped_column(default=0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    client: Mapped["Client"] = relationship("Client", back_populates="batches")
    tasks: Mapped[list["PayoutTask"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan"
    )


class PayoutTask(Base, UUIDMixin, TimestampMixin):
    """A single payout instruction within a batch."""

    __tablename__ = "payout_tasks"

    batch_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("batches.id", ondelete="CASCADE"), nullable=False
    )
    corridor_code: Mapped[str] = mapped_column(String(2), nullable=False)
    recipient_name: Mapped[str] = mapped_column(String(512), nullable=False)
    recipient_ref: Mapped[str] = mapped_column(String(255), nullable=False)  # internal ID
    amount_usd: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    preferred_asset: Mapped[AssetType | None] = mapped_column(Enum(AssetType))
    preferred_mode: Mapped[DeliveryMode | None] = mapped_column(Enum(DeliveryMode))

    # Resolved by routing engine
    resolved_asset: Mapped[AssetType | None] = mapped_column(Enum(AssetType))
    resolved_mode: Mapped[DeliveryMode | None] = mapped_column(Enum(DeliveryMode))
    partner_route: Mapped[str | None] = mapped_column(String(128))
    fallback_route: Mapped[str | None] = mapped_column(String(128))

    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus), default=TaskStatus.PENDING, nullable=False
    )
    compliance_result: Mapped[dict | None] = mapped_column(JSONB)
    partner_tx_id: Mapped[str | None] = mapped_column(String(255))
    failure_reason: Mapped[str | None] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(default=0)
    beneficiary_data: Mapped[dict] = mapped_column(JSONB, default=dict)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    batch: Mapped["Batch"] = relationship(back_populates="tasks")
    execution_events: Mapped[list] = relationship("ExecutionEvent", back_populates="task")
