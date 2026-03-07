"""
Base corridor adapter.

Each corridor (country) must implement this interface to define:
- Supported assets and delivery modes
- Required beneficiary fields
- Local payment rail configuration
- Exception handling and retry rules
- SLA targets
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CorridorConfig:
    """Static configuration for a corridor."""
    code: str                          # ISO 3166-1 alpha-2
    name: str
    usdt_enabled: bool = True
    usdc_enabled: bool = True
    wallet_enabled: bool = True
    fiat_enabled: bool = True
    min_amount_usd: float = 10.0
    max_amount_usd: float = 50_000.0
    wallet_sla_minutes: int = 60
    fiat_sla_minutes: int = 1440
    primary_wallet_partner: str = "bridge"
    primary_fiat_partner: str = ""
    fallback_partner: Optional[str] = None
    required_beneficiary_fields: list[str] = field(default_factory=list)
    kyt_risk_threshold: int = 7
    travel_rule_enabled: bool = False
    travel_rule_threshold_usd: Optional[float] = None
    notes: str = ""


@dataclass
class BeneficiaryValidationResult:
    valid: bool
    missing_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class CorridorAdapter(ABC):
    """Abstract corridor adapter — one per launch market."""

    @property
    @abstractmethod
    def config(self) -> CorridorConfig:
        """Return the static corridor configuration."""

    @abstractmethod
    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        """
        Validate that the beneficiary data satisfies corridor requirements.
        Returns a validation result with any missing fields or errors.
        """

    def get_chain_for_asset(self, asset: str) -> str:
        """Return the preferred blockchain for this corridor. Override per-country if needed."""
        return "ethereum"

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        """
        Transform generic beneficiary_data into the format expected by the
        fiat off-ramp partner for this corridor. Override per-country.
        """
        return beneficiary_data

    def get_exception_policy(self) -> dict:
        """Return the corridor-specific retry and exception policy."""
        return {
            "max_retries": 3,
            "retry_backoff_seconds": 60,
            "on_hard_fail": "cancel",
            "on_compliance_hold": "manual_review",
        }
