"""PayoutInstruction — the normalised form of a client's payout request."""
import uuid
from sqlalchemy import String, Numeric, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin, UUIDMixin


class PayoutInstruction(Base, UUIDMixin, TimestampMixin):
    """
    Normalised payout instruction after ingress validation.
    Created from raw batch item; used as the canonical record for routing.
    """

    __tablename__ = "payout_instructions"

    task_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("payout_tasks.id", ondelete="CASCADE"), nullable=False
    )
    corridor_code: Mapped[str] = mapped_column(String(2), nullable=False)

    # Amounts
    amount_usd: Mapped[float] = mapped_column(Numeric(18, 6), nullable=False)
    amount_asset: Mapped[float | None] = mapped_column(Numeric(18, 6))
    asset: Mapped[str | None] = mapped_column(String(16))  # USDT | USDC

    # Recipient
    recipient_name: Mapped[str] = mapped_column(String(512), nullable=False)
    recipient_wallet: Mapped[str | None] = mapped_column(String(255))
    recipient_bank_account: Mapped[str | None] = mapped_column(String(255))
    recipient_mobile_money: Mapped[str | None] = mapped_column(String(255))
    recipient_extra: Mapped[dict] = mapped_column(JSONB, default=dict)

    # Routing outcome
    delivery_mode: Mapped[str | None] = mapped_column(String(16))  # wallet | fiat
    partner: Mapped[str | None] = mapped_column(String(128))
    raw_payload: Mapped[dict] = mapped_column(JSONB, default=dict)
