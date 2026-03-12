"""
Brazil (BR) corridor adapter.

Key characteristics:
- High PIX adoption for fiat last mile
- Strong crypto/stablecoin usage
- Regulated by Banco Central do Brasil (BCB)
- PIX key types: CPF, CNPJ, phone, email, random key
"""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class BrazilAdapter(CorridorAdapter):

    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="BR",
            name="Brazil",
            usdt_enabled=True,
            usdc_enabled=True,
            wallet_enabled=True,
            fiat_enabled=True,
            min_amount_usd=10.0,
            max_amount_usd=50_000.0,
            wallet_sla_minutes=60,
            fiat_sla_minutes=120,       # PIX is near-instant
            primary_wallet_partner="bridge",
            primary_fiat_partner="bridge_pix",
            fallback_partner=None,
            required_beneficiary_fields=["pix_key", "recipient_document"],
            kyt_risk_threshold=7,
            travel_rule_enabled=False,
            notes="PIX is the preferred fiat rail. CPF/CNPJ required for AML.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        errors = []
        missing = []

        mode = beneficiary_data.get("mode", "wallet")
        if mode == "fiat":
            if not beneficiary_data.get("pix_key"):
                missing.append("pix_key")
            if not beneficiary_data.get("recipient_document"):
                missing.append("recipient_document")  # CPF or CNPJ

        return BeneficiaryValidationResult(
            valid=not missing and not errors,
            missing_fields=missing,
            errors=errors,
        )

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        return {
            "type": "pix",
            "pix_key": beneficiary_data.get("pix_key"),
            "pix_key_type": beneficiary_data.get("pix_key_type", "cpf"),
            "document": beneficiary_data.get("recipient_document"),
            "name": beneficiary_data.get("recipient_name"),
        }
