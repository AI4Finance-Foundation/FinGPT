"""Philippines (PH) corridor adapter — InstaPay / PESONet + GCash/Maya."""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class PhilippinesAdapter(CorridorAdapter):
    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="PH", name="Philippines",
            min_amount_usd=10.0, max_amount_usd=50_000.0,
            fiat_sla_minutes=120,           # InstaPay near-instant
            primary_fiat_partner="bridge_ph",
            required_beneficiary_fields=["bank_account_number", "bank_code"],
            notes="High remittance + crypto adoption. InstaPay for fiat. GCash mobile optional.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        missing = []
        if beneficiary_data.get("mode") == "fiat":
            for f in ["bank_account_number", "bank_code"]:
                if not beneficiary_data.get(f):
                    missing.append(f)
        return BeneficiaryValidationResult(valid=not missing, missing_fields=missing)

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        rail = "instapay" if beneficiary_data.get("instapay") else "pesonet"
        return {
            "type": rail,
            "account_number": beneficiary_data.get("bank_account_number"),
            "bank_code": beneficiary_data.get("bank_code"),
            "name": beneficiary_data.get("recipient_name"),
            "currency": "PHP",
        }
