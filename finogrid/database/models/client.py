"""Client and corridor permission models."""
import uuid
from enum import Enum as PyEnum
from sqlalchemy import String, Boolean, Enum, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin, UUIDMixin


class ClientStatus(str, PyEnum):
    PENDING_KYB = "pending_kyb"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    OFFBOARDED = "offboarded"


class KYBStatus(str, PyEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    UNDER_REVIEW = "under_review"


class Client(Base, UUIDMixin, TimestampMixin):
    """Business client registered on Finogrid."""

    __tablename__ = "clients"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    legal_name: Mapped[str] = mapped_column(String(512), nullable=False)
    registration_country: Mapped[str] = mapped_column(String(2), nullable=False)  # ISO 3166-1 alpha-2
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    status: Mapped[ClientStatus] = mapped_column(
        Enum(ClientStatus), default=ClientStatus.PENDING_KYB, nullable=False
    )
    kyb_status: Mapped[KYBStatus] = mapped_column(
        Enum(KYBStatus), default=KYBStatus.NOT_STARTED, nullable=False
    )
    kyb_provider_ref: Mapped[str | None] = mapped_column(String(255))
    webhook_url: Mapped[str | None] = mapped_column(String(1024))
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    corridor_permissions: Mapped[list["ClientCorridorPermission"]] = relationship(
        back_populates="client", cascade="all, delete-orphan"
    )
    batches: Mapped[list] = relationship("Batch", back_populates="client")


class ClientCorridorPermission(Base, UUIDMixin, TimestampMixin):
    """Per-corridor permissions and limits for a client."""

    __tablename__ = "client_corridor_permissions"

    client_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clients.id", ondelete="CASCADE"), nullable=False
    )
    corridor_code: Mapped[str] = mapped_column(String(2), nullable=False)  # ISO country code
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    max_single_payout_usd: Mapped[int] = mapped_column(Integer, default=10_000)
    max_daily_volume_usd: Mapped[int] = mapped_column(Integer, default=100_000)
    allowed_assets: Mapped[list] = mapped_column(JSONB, default=["USDT", "USDC"])
    allowed_modes: Mapped[list] = mapped_column(JSONB, default=["wallet", "fiat"])

    client: Mapped["Client"] = relationship(back_populates="corridor_permissions")
