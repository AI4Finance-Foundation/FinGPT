"""
India (IN) corridor adapter.

Key characteristics:
- UPI (Unified Payments Interface) dominant for fiat last mile
- IMPS/NEFT as fallback rails
- Regulated by RBI (Reserve Bank of India) — crypto environment evolving
- PAN or Aadhaar required for AML above threshold
- Large freelancer and contractor base
"""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class IndiaAdapter(CorridorAdapter):

    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="IN",
            name="India",
            usdt_enabled=True,
            usdc_enabled=True,
            wallet_enabled=True,
            fiat_enabled=True,
            min_amount_usd=10.0,
            max_amount_usd=50_000.0,
            wallet_sla_minutes=60,
            fiat_sla_minutes=240,         # UPI near-instant; compliance check adds time
            primary_wallet_partner="bridge",
            primary_fiat_partner="bridge_upi",
            fallback_partner=None,
            required_beneficiary_fields=["upi_id"],
            kyt_risk_threshold=7,
            travel_rule_enabled=False,
            notes="UPI is preferred fiat rail. VPA (UPI ID) is the key field.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        missing = []
        mode = beneficiary_data.get("mode", "wallet")
        if mode == "fiat":
            if not beneficiary_data.get("upi_id"):
                missing.append("upi_id")
        return BeneficiaryValidationResult(valid=not missing, missing_fields=missing)

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        return {
            "type": "upi",
            "vpa": beneficiary_data.get("upi_id"),
            "name": beneficiary_data.get("recipient_name"),
            "currency": "INR",
        }
