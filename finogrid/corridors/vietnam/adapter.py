"""Vietnam (VN) corridor adapter — bank transfer via Napas or VietQR."""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class VietnamAdapter(CorridorAdapter):
    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="VN", name="Vietnam",
            min_amount_usd=10.0, max_amount_usd=50_000.0,
            fiat_sla_minutes=480,
            primary_fiat_partner="bridge_vn",
            required_beneficiary_fields=["bank_account_number", "bank_bin_code"],
            notes="VietQR / Napas for fiat; USDT on TRON common.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        missing = []
        if beneficiary_data.get("mode") == "fiat":
            for f in ["bank_account_number", "bank_bin_code"]:
                if not beneficiary_data.get(f):
                    missing.append(f)
        return BeneficiaryValidationResult(valid=not missing, missing_fields=missing)

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        return {
            "type": "napas",
            "account_number": beneficiary_data.get("bank_account_number"),
            "bin_code": beneficiary_data.get("bank_bin_code"),
            "name": beneficiary_data.get("recipient_name"),
            "currency": "VND",
        }

    def get_chain_for_asset(self, asset: str) -> str:
        return "tron" if asset.upper() == "USDT" else "ethereum"
