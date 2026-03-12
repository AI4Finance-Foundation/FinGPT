"""Routing and compliance profile models."""
from sqlalchemy import String, Boolean, Integer, Numeric
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin, UUIDMixin


class RoutingProfile(Base, UUIDMixin, TimestampMixin):
    """Per-corridor routing configuration and partner preferences."""

    __tablename__ = "routing_profiles"

    corridor_code: Mapped[str] = mapped_column(String(2), unique=True, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Asset support
    usdt_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    usdc_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Last-mile modes
    wallet_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    fiat_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Primary and fallback partners
    wallet_partner: Mapped[str] = mapped_column(String(128), default="bridge")
    fiat_partner: Mapped[str] = mapped_column(String(128))
    fallback_partner: Mapped[str | None] = mapped_column(String(128))

    # Limits
    min_amount_usd: Mapped[float] = mapped_column(Numeric(18, 2), default=10.0)
    max_amount_usd: Mapped[float] = mapped_column(Numeric(18, 2), default=50_000.0)

    # SLA targets (minutes)
    wallet_sla_minutes: Mapped[int] = mapped_column(Integer, default=60)
    fiat_sla_minutes: Mapped[int] = mapped_column(Integer, default=1440)

    # Beneficiary data requirements per corridor
    required_beneficiary_fields: Mapped[list] = mapped_column(JSONB, default=list)

    # Retry policy
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    retry_backoff_seconds: Mapped[int] = mapped_column(Integer, default=60)

    extra_config: Mapped[dict] = mapped_column(JSONB, default=dict)


class ComplianceProfile(Base, UUIDMixin, TimestampMixin):
    """Per-corridor compliance rules and screening configuration."""

    __tablename__ = "compliance_profiles"

    corridor_code: Mapped[str] = mapped_column(String(2), unique=True, nullable=False)

    # KYT / AML
    kyt_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    kyt_risk_threshold: Mapped[int] = mapped_column(Integer, default=7)  # 0-10 scale
    sanctions_screen_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    # Travel Rule
    travel_rule_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    travel_rule_threshold_usd: Mapped[float | None] = mapped_column(Numeric(18, 2))

    # Originator / beneficiary capture
    originator_data_required: Mapped[bool] = mapped_column(Boolean, default=True)
    beneficiary_data_required: Mapped[bool] = mapped_column(Boolean, default=True)

    # Hold policy
    auto_hold_on_high_risk: Mapped[bool] = mapped_column(Boolean, default=True)
    manual_review_threshold: Mapped[int] = mapped_column(Integer, default=8)

    extra_rules: Mapped[dict] = mapped_column(JSONB, default=dict)
