"""Audit log model — append-only compliance and governance trail."""
import uuid
from sqlalchemy import String, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin, UUIDMixin


class AuditLog(Base, UUIDMixin, TimestampMixin):
    """Append-only audit log for compliance, governance, and regulatory reporting."""

    __tablename__ = "audit_logs"

    # Actor
    actor_type: Mapped[str] = mapped_column(String(64), nullable=False)  # client|agent|system|operator
    actor_id: Mapped[str | None] = mapped_column(String(255))

    # Action
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(255))

    # Context
    corridor_code: Mapped[str | None] = mapped_column(String(2))
    client_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    batch_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    task_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    # Data
    before_state: Mapped[dict | None] = mapped_column(JSONB)
    after_state: Mapped[dict | None] = mapped_column(JSONB)
    detail: Mapped[str | None] = mapped_column(Text)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    request_id: Mapped[str | None] = mapped_column(String(255))
