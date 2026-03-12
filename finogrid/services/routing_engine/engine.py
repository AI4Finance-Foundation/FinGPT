"""
Routing Engine — deterministic payout routing.

Given a PayoutTask, the engine:
1. Loads the corridor's RoutingProfile and ComplianceProfile
2. Resolves asset (USDT vs USDC)
3. Resolves delivery mode (wallet vs fiat)
4. Selects primary and fallback partners
5. Validates amount limits
6. Returns a RoutingDecision

This is intentionally deterministic — no LLM calls in this path.
"""
from __future__ import annotations

import structlog
from dataclasses import dataclass
from enum import Enum
from typing import Optional

log = structlog.get_logger()


class RoutingError(Exception):
    pass


class AssetType(str, Enum):
    USDT = "USDT"
    USDC = "USDC"


class DeliveryMode(str, Enum):
    WALLET = "wallet"
    FIAT = "fiat"


@dataclass
class RoutingDecision:
    task_id: str
    corridor_code: str
    resolved_asset: AssetType
    resolved_mode: DeliveryMode
    primary_partner: str
    fallback_partner: Optional[str]
    estimated_sla_minutes: int
    compliance_profile_id: str
    notes: list[str]


class RoutingEngine:
    """
    Stateless routing engine. Loads profiles from DB on each call.
    All decisions are logged to audit_logs.
    """

    def __init__(self, routing_profile, compliance_profile):
        self.rp = routing_profile
        self.cp = compliance_profile

    def decide(
        self,
        task_id: str,
        corridor_code: str,
        amount_usd: float,
        preferred_asset: Optional[str] = None,
        preferred_mode: Optional[str] = None,
        beneficiary_data: Optional[dict] = None,
    ) -> RoutingDecision:
        notes = []
        rp = self.rp
        beneficiary_data = beneficiary_data or {}

        # ── 1. Validate corridor is active ────────────────────────────────────
        if not rp.enabled:
            raise RoutingError(f"Corridor {corridor_code} is currently disabled")

        # ── 2. Validate amount limits ─────────────────────────────────────────
        if amount_usd < float(rp.min_amount_usd):
            raise RoutingError(
                f"Amount ${amount_usd} is below minimum ${rp.min_amount_usd} for {corridor_code}"
            )
        if amount_usd > float(rp.max_amount_usd):
            raise RoutingError(
                f"Amount ${amount_usd} exceeds maximum ${rp.max_amount_usd} for {corridor_code}"
            )

        # ── 3. Resolve asset ──────────────────────────────────────────────────
        resolved_asset = self._resolve_asset(preferred_asset, rp, notes)

        # ── 4. Resolve delivery mode ─────────────────────────────────────────
        resolved_mode = self._resolve_mode(preferred_mode, beneficiary_data, rp, notes)

        # ── 5. Select partner ─────────────────────────────────────────────────
        primary_partner, fallback_partner, sla = self._select_partner(resolved_mode, rp)

        log.info(
            "routing_decided",
            task_id=task_id,
            corridor=corridor_code,
            asset=resolved_asset,
            mode=resolved_mode,
            partner=primary_partner,
        )

        return RoutingDecision(
            task_id=task_id,
            corridor_code=corridor_code,
            resolved_asset=resolved_asset,
            resolved_mode=resolved_mode,
            primary_partner=primary_partner,
            fallback_partner=fallback_partner,
            estimated_sla_minutes=sla,
            compliance_profile_id=str(self.cp.id),
            notes=notes,
        )

    def _resolve_asset(self, preferred: Optional[str], rp, notes: list) -> AssetType:
        if preferred:
            preferred = preferred.upper()
            if preferred == "USDT" and rp.usdt_enabled:
                return AssetType.USDT
            if preferred == "USDC" and rp.usdc_enabled:
                notes.append(f"Preferred {preferred} unavailable in corridor; falling back")

        # Default priority: USDT > USDC
        if rp.usdt_enabled:
            return AssetType.USDT
        if rp.usdc_enabled:
            return AssetType.USDC

        raise RoutingError("No supported asset for this corridor")

    def _resolve_mode(
        self,
        preferred: Optional[str],
        beneficiary_data: dict,
        rp,
        notes: list,
    ) -> DeliveryMode:
        # Infer from beneficiary data if no preference
        if not preferred:
            if beneficiary_data.get("wallet"):
                preferred = "wallet"
            elif beneficiary_data.get("bank_account") or beneficiary_data.get("mobile_money"):
                preferred = "fiat"

        if preferred:
            preferred = preferred.lower()
            if preferred == "wallet" and rp.wallet_enabled:
                return DeliveryMode.WALLET
            if preferred == "fiat" and rp.fiat_enabled:
                return DeliveryMode.FIAT
            notes.append(f"Preferred mode '{preferred}' unavailable; falling back")

        # Default priority: wallet > fiat
        if rp.wallet_enabled:
            return DeliveryMode.WALLET
        if rp.fiat_enabled:
            return DeliveryMode.FIAT

        raise RoutingError("No supported delivery mode for this corridor")

    def _select_partner(self, mode: DeliveryMode, rp) -> tuple[str, Optional[str], int]:
        if mode == DeliveryMode.WALLET:
            return rp.wallet_partner, rp.fallback_partner, rp.wallet_sla_minutes
        return rp.fiat_partner, rp.fallback_partner, rp.fiat_sla_minutes
