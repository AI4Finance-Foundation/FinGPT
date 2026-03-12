"""Indonesia (ID) corridor adapter — BI-FAST / QRIS for fiat."""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class IndonesiaAdapter(CorridorAdapter):
    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="ID", name="Indonesia",
            min_amount_usd=10.0, max_amount_usd=50_000.0,
            fiat_sla_minutes=240,
            primary_fiat_partner="bridge_id",
            required_beneficiary_fields=["bank_account_number", "bank_code", "nik"],
            notes="BI-FAST preferred. NIK (national ID) required for AML.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        missing = []
        if beneficiary_data.get("mode") == "fiat":
            for f in ["bank_account_number", "bank_code"]:
                if not beneficiary_data.get(f):
                    missing.append(f)
        return BeneficiaryValidationResult(valid=not missing, missing_fields=missing)

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        return {
            "type": "bi_fast",
            "account_number": beneficiary_data.get("bank_account_number"),
            "bank_code": beneficiary_data.get("bank_code"),
            "nik": beneficiary_data.get("nik"),
            "name": beneficiary_data.get("recipient_name"),
            "currency": "IDR",
        }
