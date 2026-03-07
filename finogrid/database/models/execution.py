"""Execution event model — immutable audit trail of every partner interaction."""
import uuid
from enum import Enum as PyEnum
from sqlalchemy import String, Enum, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin, UUIDMixin


class EventType(str, PyEnum):
    TASK_CREATED = "task_created"
    ROUTING_RESOLVED = "routing_resolved"
    COMPLIANCE_PASS = "compliance_pass"
    COMPLIANCE_HOLD = "compliance_hold"
    COMPLIANCE_FAIL = "compliance_fail"
    PARTNER_SUBMITTED = "partner_submitted"
    PARTNER_STATUS_UPDATE = "partner_status_update"
    PARTNER_COMPLETED = "partner_completed"
    PARTNER_FAILED = "partner_failed"
    RETRY_INITIATED = "retry_initiated"
    MANUAL_REVIEW_STARTED = "manual_review_started"
    MANUAL_REVIEW_RESOLVED = "manual_review_resolved"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"


class ExecutionEvent(Base, UUIDMixin, TimestampMixin):
    """Immutable log of every state transition for a payout task."""

    __tablename__ = "execution_events"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("payout_tasks.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[EventType] = mapped_column(Enum(EventType), nullable=False)
    partner: Mapped[str | None] = mapped_column(String(128))
    partner_ref: Mapped[str | None] = mapped_column(String(255))
    payload: Mapped[dict] = mapped_column(JSONB, default=dict)
    note: Mapped[str | None] = mapped_column(Text)

    task: Mapped["PayoutTask"] = relationship("PayoutTask", back_populates="execution_events")
