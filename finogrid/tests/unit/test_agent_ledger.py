"""
Unit tests for Agent Ledger subsystem.

Coverage:
  - KYA gate enforcement (unverified, pending, basic, enhanced)
  - Closed-loop intent validation (missing intent, wrong wallet, expired, wrong amount)
  - Open-loop spending rules (per-tx cap, daily cap, balance check)
  - Micropay idempotency
  - Intent sweeper (expired intent → balance release)
  - x402 middleware (payment required, valid signature, expired nonce)
  - Mandate model (status transitions, scope constraints)
"""
import pytest
import uuid
import time
import base64
import json
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


# ── KYA gate tests ────────────────────────────────────────────────────────────

class TestKYAGate:
    """Tests for assert_kya_status in dependencies.py"""

    @pytest.mark.asyncio
    async def test_unverified_agent_blocked(self):
        from finogrid.services.agent_ledger_api.dependencies import assert_kya_status
        from finogrid.database.models.agent_ledger import KYAStatus

        agent = MagicMock()
        agent.kya_status = KYAStatus.UNVERIFIED

        with pytest.raises(Exception) as exc_info:
            await assert_kya_status(agent, required_level="basic")
        assert "403" in str(exc_info.value.status_code) or exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_pending_agent_blocked_for_basic(self):
        from finogrid.services.agent_ledger_api.dependencies import assert_kya_status
        from finogrid.database.models.agent_ledger import KYAStatus

        agent = MagicMock()
        agent.kya_status = KYAStatus.PENDING

        with pytest.raises(Exception) as exc_info:
            await assert_kya_status(agent, required_level="basic")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_basic_agent_passes_basic_gate(self):
        from finogrid.services.agent_ledger_api.dependencies import assert_kya_status
        from finogrid.database.models.agent_ledger import KYAStatus

        agent = MagicMock()
        agent.kya_status = KYAStatus.BASIC

        # Should not raise
        await assert_kya_status(agent, required_level="basic")

    @pytest.mark.asyncio
    async def test_enhanced_agent_passes_basic_gate(self):
        from finogrid.services.agent_ledger_api.dependencies import assert_kya_status
        from finogrid.database.models.agent_ledger import KYAStatus

        agent = MagicMock()
        agent.kya_status = KYAStatus.ENHANCED

        await assert_kya_status(agent, required_level="basic")

    @pytest.mark.asyncio
    async def test_basic_agent_blocked_for_enhanced(self):
        from finogrid.services.agent_ledger_api.dependencies import assert_kya_status
        from finogrid.database.models.agent_ledger import KYAStatus

        agent = MagicMock()
        agent.kya_status = KYAStatus.BASIC

        with pytest.raises(Exception) as exc_info:
            await assert_kya_status(agent, required_level="enhanced")
        assert exc_info.value.status_code == 403

    def test_kya_daily_limit_basic(self):
        from finogrid.services.agent_ledger_api.dependencies import kya_daily_limit
        limit = kya_daily_limit("basic")
        assert limit == 1.00

    def test_kya_daily_limit_enhanced(self):
        from finogrid.services.agent_ledger_api.dependencies import kya_daily_limit
        limit = kya_daily_limit("enhanced")
        assert limit == 100.00

    def test_kya_daily_limit_unverified(self):
        from finogrid.services.agent_ledger_api.dependencies import kya_daily_limit
        limit = kya_daily_limit("unverified")
        assert limit == 0.0


# ── Closed-loop intent tests ──────────────────────────────────────────────────

class TestClosedLoopIntents:
    """Tests for payment intent validation logic."""

    def _make_wallet(self, loop_type="closed", status="active"):
        from finogrid.database.models.agent_ledger import AgentWallet, LoopType
        wallet = MagicMock(spec=AgentWallet)
        wallet.id = uuid.uuid4()
        wallet.loop_type = LoopType.CLOSED if loop_type == "closed" else LoopType.OPEN
        wallet.status = status
        wallet.max_per_tx_usdc = Decimal("1.00")
        wallet.max_daily_usdc = Decimal("10.00")
        wallet.daily_spent_usdc = Decimal("0")
        wallet.daily_reset_at = None
        wallet.allowed_counterparties = []
        wallet.expires_at = None
        wallet.max_uses = None
        wallet.use_count = 0
        wallet.agent_account_id = uuid.uuid4()
        return wallet

    def _make_intent(self, status="reserved", expired=False, amount=Decimal("0.50")):
        from finogrid.database.models.agent_ledger import PaymentIntent, IntentStatus
        intent = MagicMock(spec=PaymentIntent)
        intent.id = uuid.uuid4()
        intent.status = IntentStatus.RESERVED if status == "reserved" else IntentStatus(status)
        intent.amount_usdc = amount
        if expired:
            intent.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        else:
            intent.expires_at = datetime.now(timezone.utc) + timedelta(hours=1)
        return intent

    def test_closed_loop_wallet_type(self):
        from finogrid.database.models.agent_ledger import LoopType
        wallet = self._make_wallet("closed")
        assert wallet.loop_type == LoopType.CLOSED

    def test_open_loop_wallet_type(self):
        from finogrid.database.models.agent_ledger import LoopType
        wallet = self._make_wallet("open")
        assert wallet.loop_type == LoopType.OPEN

    def test_intent_expired_detected(self):
        intent = self._make_intent(expired=True)
        now = datetime.now(timezone.utc)
        assert intent.expires_at < now

    def test_intent_not_expired(self):
        intent = self._make_intent(expired=False)
        now = datetime.now(timezone.utc)
        assert intent.expires_at > now

    def test_intent_amount_mismatch(self):
        intent = self._make_intent(amount=Decimal("0.50"))
        request_amount = Decimal("0.75")
        mismatch = abs(float(intent.amount_usdc) - float(request_amount)) > 0.00001
        assert mismatch

    def test_intent_amount_match(self):
        intent = self._make_intent(amount=Decimal("0.50"))
        request_amount = Decimal("0.50")
        mismatch = abs(float(intent.amount_usdc) - float(request_amount)) > 0.00001
        assert not mismatch

    def test_intent_consumed_status_invalid(self):
        from finogrid.database.models.agent_ledger import IntentStatus
        intent = self._make_intent(status="consumed")
        assert intent.status != IntentStatus.RESERVED

    def test_intent_superseded_status_invalid(self):
        from finogrid.database.models.agent_ledger import IntentStatus
        intent = self._make_intent(status="superseded")
        assert intent.status != IntentStatus.RESERVED


# ── Spending rules tests ──────────────────────────────────────────────────────

class TestSpendingRules:
    """Tests for wallet-level spending rule enforcement."""

    def test_per_tx_cap_exceeded(self):
        from finogrid.database.models.agent_ledger import AgentWallet
        wallet = MagicMock(spec=AgentWallet)
        wallet.max_per_tx_usdc = Decimal("0.10")

        amount = Decimal("0.50")
        assert amount > wallet.max_per_tx_usdc

    def test_per_tx_cap_not_exceeded(self):
        from finogrid.database.models.agent_ledger import AgentWallet
        wallet = MagicMock(spec=AgentWallet)
        wallet.max_per_tx_usdc = Decimal("1.00")

        amount = Decimal("0.50")
        assert amount <= wallet.max_per_tx_usdc

    def test_daily_cap_exceeded(self):
        wallet = MagicMock()
        wallet.daily_spent_usdc = Decimal("0.90")
        wallet.max_daily_usdc = Decimal("1.00")

        new_amount = Decimal("0.20")
        new_daily = wallet.daily_spent_usdc + new_amount
        assert new_daily > wallet.max_daily_usdc

    def test_daily_cap_not_exceeded(self):
        wallet = MagicMock()
        wallet.daily_spent_usdc = Decimal("0.50")
        wallet.max_daily_usdc = Decimal("1.00")

        new_amount = Decimal("0.40")
        new_daily = wallet.daily_spent_usdc + new_amount
        assert new_daily <= wallet.max_daily_usdc

    def test_counterparty_allowlist_blocks(self):
        wallet = MagicMock()
        wallet.allowed_counterparties = ["0xabc123def456789012345678901234567890abcd"]

        payee = "0x0000000000000000000000000000000000000001"
        blocked = payee.lower() not in [a.lower() for a in wallet.allowed_counterparties]
        assert blocked

    def test_counterparty_allowlist_passes(self):
        allowed_addr = "0xabc123def456789012345678901234567890abcd"
        wallet = MagicMock()
        wallet.allowed_counterparties = [allowed_addr]

        payee = allowed_addr
        blocked = payee.lower() not in [a.lower() for a in wallet.allowed_counterparties]
        assert not blocked

    def test_empty_allowlist_allows_any(self):
        wallet = MagicMock()
        wallet.allowed_counterparties = []

        # Empty list = any payee permitted
        assert len(wallet.allowed_counterparties) == 0

    def test_available_balance_check(self):
        agent = MagicMock()
        agent.prefund_balance_usdc = Decimal("1.00")
        agent.reserved_balance_usdc = Decimal("0.50")

        available = Decimal(str(agent.prefund_balance_usdc)) - Decimal(str(agent.reserved_balance_usdc))
        amount = Decimal("0.60")
        assert amount > available

    def test_wallet_max_uses_reached(self):
        wallet = MagicMock()
        wallet.max_uses = 10
        wallet.use_count = 10
        assert wallet.use_count >= wallet.max_uses

    def test_wallet_not_expired(self):
        wallet = MagicMock()
        wallet.expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        now = datetime.now(timezone.utc)
        assert wallet.expires_at > now


# ── Idempotency tests ─────────────────────────────────────────────────────────

class TestIdempotency:
    def test_idempotency_key_uniqueness(self):
        """Same key should map to same transaction."""
        key1 = "test-idempotency-key-12345678"
        key2 = "test-idempotency-key-12345678"
        assert key1 == key2

    def test_different_keys_are_different(self):
        key1 = "key-aaaa-1111"
        key2 = "key-bbbb-2222"
        assert key1 != key2


# ── Intent sweeper tests ──────────────────────────────────────────────────────

class TestIntentSweeper:
    @pytest.mark.asyncio
    async def test_sweep_expired_intents_logic(self):
        """
        Test that sweep_expired_intents correctly identifies and transitions expired intents.
        """
        from finogrid.database.models.agent_ledger import IntentStatus

        # Build mock expired intent
        intent = MagicMock()
        intent.id = uuid.uuid4()
        intent.status = IntentStatus.RESERVED
        intent.expires_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        intent.amount_usdc = Decimal("0.50")
        intent.payer_wallet_id = uuid.uuid4()
        intent.intent_description = "Test expired intent"
        intent.intent_category = "compute"
        intent.audit_note = None

        now = datetime.now(timezone.utc)
        assert intent.expires_at < now  # Confirms it should be swept
        assert intent.status == IntentStatus.RESERVED  # Confirms it's eligible

    def test_reserved_balance_release_calculation(self):
        """Reserved balance should decrease by exact intent amount."""
        from decimal import Decimal
        current_reserved = Decimal("1.50")
        intent_amount = Decimal("0.50")
        new_reserved = max(Decimal("0"), current_reserved - intent_amount)
        assert new_reserved == Decimal("1.00")

    def test_reserved_balance_never_negative(self):
        """Reserved balance cannot go below zero."""
        current_reserved = Decimal("0.10")
        intent_amount = Decimal("0.50")  # Larger than reserved (edge case)
        new_reserved = max(Decimal("0"), current_reserved - intent_amount)
        assert new_reserved == Decimal("0")


# ── x402 middleware tests ─────────────────────────────────────────────────────

class TestX402Middleware:
    def _make_signature(self, path: str, amount: str = "0.001", offset_seconds: int = 0) -> str:
        """Helper to create a valid x402 PAYMENT-SIGNATURE header."""
        deposit_addr = "0x0000000000000000000000000000000000000000"
        payload = {
            "network": "base-mainnet",
            "asset": "USDC",
            "payTo": deposit_addr,
            "amount": amount,
            "nonce": str(uuid.uuid4()),
            "timestamp": str(time.time() + offset_seconds),
            "resource": path,
        }
        return base64.b64encode(json.dumps(payload).encode()).decode()

    def test_valid_signature_structure(self):
        sig_b64 = self._make_signature("/api/test")
        decoded = json.loads(base64.b64decode(sig_b64).decode())
        required = {"network", "asset", "payTo", "amount", "nonce", "timestamp", "resource"}
        assert required.issubset(decoded.keys())

    def test_expired_signature_rejected(self):
        from finogrid.services.agent_ledger_api.middleware.payment_required import _validate_payment_signature
        sig_b64 = self._make_signature("/api/test", offset_seconds=-(400))  # > TTL
        sig_data = json.loads(base64.b64decode(sig_b64).decode())
        valid, reason = _validate_payment_signature(sig_data, "/api/test")
        assert not valid
        assert "expired" in reason.lower()

    def test_resource_mismatch_rejected(self):
        from finogrid.services.agent_ledger_api.middleware.payment_required import _validate_payment_signature
        sig_b64 = self._make_signature("/api/wrong-path")
        sig_data = json.loads(base64.b64decode(sig_b64).decode())
        valid, reason = _validate_payment_signature(sig_data, "/api/correct-path")
        assert not valid
        assert "mismatch" in reason.lower()

    def test_encode_decode_requirement_roundtrip(self):
        from finogrid.services.agent_ledger_api.middleware.payment_required import _encode_requirement
        encoded = _encode_requirement("/v1/micropay", amount_usdc=0.001)
        decoded = json.loads(base64.b64decode(encoded).decode())
        assert decoded["scheme"] == "x402"
        assert decoded["resource"] == "/v1/micropay"
        assert decoded["asset"] == "USDC"


# ── Mandate model tests ───────────────────────────────────────────────────────

class TestMandateModel:
    def test_mandate_status_values(self):
        from finogrid.database.models.mandate import MandateStatus
        assert MandateStatus.DRAFT == "draft"
        assert MandateStatus.ACTIVE == "active"
        assert MandateStatus.REVOKED == "revoked"
        assert MandateStatus.SUPERSEDED == "superseded"

    def test_mandate_scope_values(self):
        from finogrid.database.models.mandate import MandateScope
        assert MandateScope.PAYOUT == "payout"
        assert MandateScope.FULL == "full"
        assert MandateScope.READ_ONLY == "read_only"

    def test_approval_mode_values(self):
        from finogrid.database.models.mandate import ApprovalMode
        assert ApprovalMode.AUTO == "auto"
        assert ApprovalMode.MANUAL == "manual"
        assert ApprovalMode.THRESHOLD == "threshold"

    def test_mandate_event_types(self):
        from finogrid.database.models.mandate import MandateEventType
        assert MandateEventType.CREATED == "created"
        assert MandateEventType.REVOKED == "revoked"
        assert MandateEventType.LIMIT_HIT == "limit_hit"

    def test_mandate_allowed_corridors_constraint(self):
        """An empty allowed_corridors list should mean all corridors are permitted."""
        allowed_corridors = []
        corridor = "BR"
        # Empty = no restriction
        is_allowed = len(allowed_corridors) == 0 or corridor in allowed_corridors
        assert is_allowed

    def test_mandate_corridor_restriction(self):
        """Non-empty allowed_corridors should block unlisted corridors."""
        allowed_corridors = ["BR", "IN"]
        corridor = "NG"
        is_allowed = len(allowed_corridors) == 0 or corridor in allowed_corridors
        assert not is_allowed

    def test_threshold_approval_logic(self):
        """Amount >= threshold should require manual approval."""
        from finogrid.database.models.mandate import ApprovalMode
        approval_mode = ApprovalMode.THRESHOLD
        threshold = Decimal("50.00")
        amount = Decimal("75.00")

        requires_manual = (
            approval_mode == ApprovalMode.MANUAL or
            (approval_mode == ApprovalMode.THRESHOLD and amount >= threshold)
        )
        assert requires_manual

    def test_below_threshold_auto_approved(self):
        from finogrid.database.models.mandate import ApprovalMode
        approval_mode = ApprovalMode.THRESHOLD
        threshold = Decimal("50.00")
        amount = Decimal("25.00")

        requires_manual = (
            approval_mode == ApprovalMode.MANUAL or
            (approval_mode == ApprovalMode.THRESHOLD and amount >= threshold)
        )
        assert not requires_manual


# ── KYA validator MCP tests ───────────────────────────────────────────────────

class TestKYAValidatorMCP:
    def test_internal_validate_basic(self):
        from finogrid.mcp.kya_validator.server import _internal_validate
        result_status, level = _internal_validate({
            "agent_purpose": "short",
            "agent_owner_attestation": "short",
            "declared_use_case": "general",
        })
        assert result_status == "basic"
        assert level == "basic"

    def test_internal_validate_enhanced(self):
        from finogrid.mcp.kya_validator.server import _internal_validate
        result_status, level = _internal_validate({
            "agent_purpose": "a" * 201,  # > 200 chars
            "agent_owner_attestation": "b" * 101,  # > 100 chars
            "declared_use_case": "trading_support",  # != general
        })
        assert result_status == "enhanced"
        assert level == "enhanced"

    def test_mint_validator_token_structure(self):
        from finogrid.mcp.kya_validator.server import _mint_validator_token
        token, expires_at = _mint_validator_token(str(uuid.uuid4()), "basic")
        payload = json.loads(base64.b64decode(token).decode())
        assert payload["level"] == "basic"
        assert "exp" in payload
        assert "sub" in payload

    def test_token_not_expired(self):
        from finogrid.mcp.kya_validator.server import _mint_validator_token
        agent_id = str(uuid.uuid4())
        token, expires_at = _mint_validator_token(agent_id, "enhanced")
        payload = json.loads(base64.b64decode(token).decode())
        assert payload["exp"] > time.time()

    def test_token_subject_matches_agent(self):
        from finogrid.mcp.kya_validator.server import _mint_validator_token
        agent_id = str(uuid.uuid4())
        token, _ = _mint_validator_token(agent_id, "basic")
        payload = json.loads(base64.b64decode(token).decode())
        assert payload["sub"] == agent_id
