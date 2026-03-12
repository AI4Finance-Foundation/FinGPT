"""Unit tests for the Routing Engine."""
import pytest
from unittest.mock import MagicMock
from finogrid.services.routing_engine.engine import (
    RoutingEngine, RoutingDecision, RoutingError, AssetType, DeliveryMode
)


def make_profiles(
    usdt=True, usdc=True, wallet=True, fiat=True,
    min_usd=10.0, max_usd=50_000.0,
    wallet_partner="bridge", fiat_partner="bridge_pix",
    wallet_sla=60, fiat_sla=1440,
):
    rp = MagicMock()
    rp.enabled = True
    rp.usdt_enabled = usdt
    rp.usdc_enabled = usdc
    rp.wallet_enabled = wallet
    rp.fiat_enabled = fiat
    rp.min_amount_usd = min_usd
    rp.max_amount_usd = max_usd
    rp.wallet_partner = wallet_partner
    rp.fiat_partner = fiat_partner
    rp.fallback_partner = None
    rp.wallet_sla_minutes = wallet_sla
    rp.fiat_sla_minutes = fiat_sla

    cp = MagicMock()
    cp.id = "test-profile-id"

    return rp, cp


def test_basic_routing_usdt_wallet():
    rp, cp = make_profiles()
    engine = RoutingEngine(rp, cp)
    decision = engine.decide("task-1", "BR", 100.0)
    assert decision.resolved_asset == AssetType.USDT
    assert decision.resolved_mode == DeliveryMode.WALLET
    assert decision.primary_partner == "bridge"


def test_fiat_mode_from_preferred():
    rp, cp = make_profiles()
    engine = RoutingEngine(rp, cp)
    decision = engine.decide("task-2", "BR", 500.0, preferred_mode="fiat")
    assert decision.resolved_mode == DeliveryMode.FIAT
    assert decision.primary_partner == "bridge_pix"


def test_usdc_preferred():
    rp, cp = make_profiles()
    engine = RoutingEngine(rp, cp)
    decision = engine.decide("task-3", "BR", 100.0, preferred_asset="USDC")
    assert decision.resolved_asset == AssetType.USDC


def test_amount_below_minimum_raises():
    rp, cp = make_profiles(min_usd=50.0)
    engine = RoutingEngine(rp, cp)
    with pytest.raises(RoutingError, match="below minimum"):
        engine.decide("task-4", "BR", 5.0)


def test_amount_above_maximum_raises():
    rp, cp = make_profiles(max_usd=1000.0)
    engine = RoutingEngine(rp, cp)
    with pytest.raises(RoutingError, match="exceeds maximum"):
        engine.decide("task-5", "BR", 2000.0)


def test_disabled_corridor_raises():
    rp, cp = make_profiles()
    rp.enabled = False
    engine = RoutingEngine(rp, cp)
    with pytest.raises(RoutingError, match="disabled"):
        engine.decide("task-6", "BR", 100.0)


def test_mode_inferred_from_wallet_address():
    rp, cp = make_profiles()
    engine = RoutingEngine(rp, cp)
    decision = engine.decide(
        "task-7", "NG", 100.0,
        beneficiary_data={"wallet": "TXxxx"}
    )
    assert decision.resolved_mode == DeliveryMode.WALLET


def test_mode_inferred_from_bank_account():
    rp, cp = make_profiles()
    engine = RoutingEngine(rp, cp)
    decision = engine.decide(
        "task-8", "IN", 100.0,
        beneficiary_data={"bank_account": "12345678"}
    )
    assert decision.resolved_mode == DeliveryMode.FIAT


def test_no_asset_available_raises():
    rp, cp = make_profiles(usdt=False, usdc=False)
    engine = RoutingEngine(rp, cp)
    with pytest.raises(RoutingError, match="No supported asset"):
        engine.decide("task-9", "BR", 100.0)
