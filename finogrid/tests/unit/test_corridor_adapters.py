"""Unit tests for corridor adapter validation logic."""
import pytest
from finogrid.corridors import get_adapter


@pytest.mark.parametrize("code", ["BR", "AR", "VN", "IN", "AE", "ID", "PH", "NG"])
def test_all_corridors_registered(code):
    adapter = get_adapter(code)
    assert adapter is not None
    assert adapter.config.code == code


def test_brazil_fiat_requires_pix_key():
    adapter = get_adapter("BR")
    result = adapter.validate_beneficiary({"mode": "fiat"})
    assert not result.valid
    assert "pix_key" in result.missing_fields


def test_brazil_wallet_no_extra_fields():
    adapter = get_adapter("BR")
    result = adapter.validate_beneficiary({"mode": "wallet", "wallet": "0xabc"})
    assert result.valid


def test_nigeria_fiat_requires_bvn_above_threshold():
    adapter = get_adapter("NG")
    result = adapter.validate_beneficiary({
        "mode": "fiat",
        "bank_account_number": "1234567890",
        "bank_code": "044",
        "amount_usd": 500,
        # bvn missing
    })
    assert not result.valid
    assert "bvn" in result.missing_fields


def test_nigeria_uses_tron_for_usdt():
    adapter = get_adapter("NG")
    assert adapter.get_chain_for_asset("USDT") == "tron"
    assert adapter.get_chain_for_asset("USDC") == "ethereum"


def test_india_fiat_requires_upi_id():
    adapter = get_adapter("IN")
    result = adapter.validate_beneficiary({"mode": "fiat"})
    assert not result.valid
    assert "upi_id" in result.missing_fields


def test_uae_has_higher_max_limit():
    adapter = get_adapter("AE")
    assert adapter.config.max_amount_usd == 100_000.0


def test_unsupported_corridor_raises():
    with pytest.raises(ValueError):
        get_adapter("US")
