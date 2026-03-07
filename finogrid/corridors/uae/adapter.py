"""UAE (AE) corridor adapter — regional hub with IBAN bank transfers."""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class UAEAdapter(CorridorAdapter):
    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="AE", name="UAE",
            min_amount_usd=10.0, max_amount_usd=100_000.0,  # Higher limit for hub market
            fiat_sla_minutes=480,
            primary_fiat_partner="bridge_ae",
            required_beneficiary_fields=["iban", "swift_bic"],
            kyt_risk_threshold=6,
            notes="IBAN required. UAE is a regional hub — higher limits. VARA regulated.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        missing = []
        if beneficiary_data.get("mode") == "fiat":
            for f in ["iban", "swift_bic"]:
                if not beneficiary_data.get(f):
                    missing.append(f)
        return BeneficiaryValidationResult(valid=not missing, missing_fields=missing)

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        return {
            "type": "swift_iban",
            "iban": beneficiary_data.get("iban"),
            "swift_bic": beneficiary_data.get("swift_bic"),
            "name": beneficiary_data.get("recipient_name"),
            "currency": "AED",
        }
