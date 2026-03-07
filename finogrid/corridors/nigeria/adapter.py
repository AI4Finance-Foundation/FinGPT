"""
Nigeria (NG) corridor adapter.

Key characteristics:
- Largest stablecoin market in Sub-Saharan Africa
- Top-tier global stablecoin adoption ranking
- Strong remittance and cross-border usage
- Regulated by CBN (Central Bank of Nigeria) and SEC Nigeria
- Last mile: bank transfer (NIBSS) or mobile money (OPay, PalmPay)
- BVN (Bank Verification Number) required for AML
"""
from ..base import CorridorAdapter, CorridorConfig, BeneficiaryValidationResult


class NigeriaAdapter(CorridorAdapter):

    @property
    def config(self) -> CorridorConfig:
        return CorridorConfig(
            code="NG",
            name="Nigeria",
            usdt_enabled=True,
            usdc_enabled=True,
            wallet_enabled=True,
            fiat_enabled=True,
            min_amount_usd=10.0,
            max_amount_usd=50_000.0,
            wallet_sla_minutes=60,
            fiat_sla_minutes=1440,
            primary_wallet_partner="bridge",
            primary_fiat_partner="bridge_ng",
            fallback_partner=None,
            required_beneficiary_fields=["bank_account_number", "bank_code", "bvn"],
            kyt_risk_threshold=6,
            travel_rule_enabled=False,
            notes="Nigeria modeled as standalone corridor, not 'Africa'. "
                  "BVN mandatory for NGN fiat delivery per CBN AML rules.",
        )

    def validate_beneficiary(self, beneficiary_data: dict) -> BeneficiaryValidationResult:
        missing = []
        mode = beneficiary_data.get("mode", "wallet")
        if mode == "fiat":
            for field in ["bank_account_number", "bank_code"]:
                if not beneficiary_data.get(field):
                    missing.append(field)
            # BVN required per CBN AML rules for transfers above threshold
            amount = float(beneficiary_data.get("amount_usd", 0))
            if amount >= 100 and not beneficiary_data.get("bvn"):
                missing.append("bvn")
        return BeneficiaryValidationResult(valid=not missing, missing_fields=missing)

    def format_bank_payload(self, beneficiary_data: dict) -> dict:
        return {
            "type": "nibss",
            "account_number": beneficiary_data.get("bank_account_number"),
            "bank_code": beneficiary_data.get("bank_code"),
            "bvn": beneficiary_data.get("bvn"),
            "name": beneficiary_data.get("recipient_name"),
            "currency": "NGN",
        }

    def get_chain_for_asset(self, asset: str) -> str:
        # TRON (TRC-20 USDT) widely used in Nigeria due to low fees
        if asset.upper() == "USDT":
            return "tron"
        return "ethereum"
