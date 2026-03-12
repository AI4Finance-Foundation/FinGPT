"""
Argentina (AR) corridor adapter.

Key characteristics:
- Extremely high stablecoin adoption driven by USD access demand
- Informal dollar-pegged economy — USDT is widely preferred over USDC
- ARS is volatile; recipients often prefer stablecoin wallet delivery
- Banco Central de la República Argentina (BCRA) restrictions on FX
"""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class ArgentinaAdapter(CorridorAdapter):

    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="AR",
            name="Argentina",
            usdt_enabled=True,
            usdc_enabled=True,
            wallet_enabled=True,
            fiat_enabled=True,
            min_amount_usd=10.0,
            max_amount_usd=25_000.0,       # Lower limit due to FX regulations
            wallet_sla_minutes=60,
            fiat_sla_minutes=2880,         # Fiat settlement slower due to controls
            primary_wallet_partner="bridge",
            primary_fiat_partner="bridge_ar",
            fallback_partner=None,
            required_beneficiary_fields=["cbu_or_alias", "cuit"],
            kyt_risk_threshold=6,           # Stricter due to regulatory environment
            travel_rule_enabled=False,
            notes="USDT preferred; wallet delivery is primary mode. ARS fiat limited.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        missing = []
        mode = beneficiary_data.get("mode", "wallet")
        if mode == "fiat":
            if not beneficiary_data.get("cbu_or_alias"):
                missing.append("cbu_or_alias")
            if not beneficiary_data.get("cuit"):
                missing.append("cuit")
        return BeneficiaryValidationResult(valid=not missing, missing_fields=missing)

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        return {
            "type": "bank_transfer_ar",
            "cbu_or_alias": beneficiary_data.get("cbu_or_alias"),
            "cuit": beneficiary_data.get("cuit"),
            "name": beneficiary_data.get("recipient_name"),
        }
