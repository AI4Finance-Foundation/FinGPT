"""Pydantic schemas for Agent Ledger API."""
import uuid
from decimal import Decimal
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


# ── Agent Accounts ─────────────────────────────────────────────────────────────

class AgentAccountCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    owner_client_id: Optional[uuid.UUID] = None
    chain: Literal["base", "polygon"] = "base"
    metadata: dict = Field(default_factory=dict)


class AgentAccountCreateResponse(BaseModel):
    agent_account_id: uuid.UUID
    api_key: str  # Returned ONCE — never stored in plaintext
    status: str = "active"
    kya_status: str = "unverified"
    chain: str
    message: str = (
        "Agent registered. Submit KYA at POST /v1/agent-accounts/{id}/kya "
        "before making outbound payments."
    )


class AgentBalanceResponse(BaseModel):
    agent_account_id: uuid.UUID
    kya_status: str
    prefund_balance_usdc: Decimal
    reserved_balance_usdc: Decimal
    available_balance_usdc: Decimal  # prefund - reserved
    chain: str
    recent_entries: list[dict] = []


# ── KYA ───────────────────────────────────────────────────────────────────────

class KYASubmitRequest(BaseModel):
    agent_purpose: str = Field(..., min_length=10, max_length=2000)
    declared_use_case: Literal[
        "data_retrieval", "content_generation", "trading_support", "general"
    ]
    agent_owner_attestation: str = Field(
        ...,
        min_length=20,
        description="Owner's signed statement that they control this agent and accept liability",
    )
    validator_name: Optional[str] = Field(
        None,
        description="Preferred validator: sardine | persona | chainalysis | internal",
    )


class KYAStatusResponse(BaseModel):
    agent_account_id: uuid.UUID
    kya_status: str
    validator_name: Optional[str]
    validator_ref: Optional[str]
    validator_token_present: bool  # never expose the token itself
    validator_expires_at: Optional[str]
    validated_at: Optional[str]
    message: str


# ── Agent Wallets ─────────────────────────────────────────────────────────────

class AgentWalletCreateRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=255)
    wallet_address: str = Field(..., description="EVM address (0x...)")
    loop_type: Literal["open", "closed"] = Field(
        ...,
        description="Owner sets this at creation; immutable after. "
                    "closed = PaymentIntent required for every payment.",
    )
    max_per_tx_usdc: Decimal = Field(default=Decimal("0.10"), gt=0)
    max_daily_usdc: Decimal = Field(default=Decimal("1.00"), gt=0)
    allowed_counterparties: list[str] = Field(
        default_factory=list,
        description="Empty = any payee. Non-empty = strict allowlist.",
    )
    expires_at: Optional[datetime] = None
    max_uses: Optional[int] = Field(None, gt=0)

    @field_validator("wallet_address")
    @classmethod
    def validate_evm_address(cls, v: str) -> str:
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError("wallet_address must be a valid EVM address (0x + 40 hex chars)")
        return v.lower()


class AgentWalletCreateResponse(BaseModel):
    wallet_id: uuid.UUID
    wallet_address: str
    loop_type: str
    spending_rules: dict
    status: str = "active"


# ── Payment Intents ───────────────────────────────────────────────────────────

class PaymentIntentCreateRequest(BaseModel):
    payer_wallet_id: uuid.UUID
    amount_usdc: Decimal = Field(..., gt=0, description="Amount to reserve")
    intent_description: str = Field(
        ..., min_length=10,
        description="Human-readable purpose of this payment",
    )
    intent_category: Literal[
        "compute", "data", "agent_service", "content", "offramp", "other"
    ] = "other"
    expires_at: datetime = Field(
        ..., description="When this intent auto-expires if unused"
    )


class PaymentIntentCreateResponse(BaseModel):
    payment_intent_id: uuid.UUID
    payer_wallet_id: uuid.UUID
    amount_usdc: Decimal
    intent_category: str
    status: str = "reserved"
    expires_at: str
    message: str = "Intent reserved. Use payment_intent_id in POST /v1/micropay."


class PaymentIntentSupersede(BaseModel):
    new_intent_description: str = Field(..., min_length=10)
    new_intent_category: Literal[
        "compute", "data", "agent_service", "content", "offramp", "other"
    ]
    new_amount_usdc: Decimal = Field(..., gt=0)
    new_expires_at: datetime
    audit_note: str = Field(
        ..., min_length=10,
        description="Required explanation of why the intent changed",
    )


# ── Micropay ──────────────────────────────────────────────────────────────────

class MicroPayRequest(BaseModel):
    idempotency_key: str = Field(
        ..., min_length=8, max_length=255,
        description="Caller-supplied unique key; prevents double-spend on retries",
    )
    payer_wallet_id: uuid.UUID
    payee_address: str = Field(..., description="On-chain EVM address of payee")
    amount_usdc: Decimal = Field(..., gt=0)
    payment_intent_id: Optional[uuid.UUID] = Field(
        None,
        description="Required for closed-loop wallets. Must reference a 'reserved' intent.",
    )
    x402_payment_header: Optional[str] = None
    x402_resource_url: Optional[str] = None
    metadata: dict = Field(default_factory=dict)

    @field_validator("payee_address")
    @classmethod
    def validate_payee(cls, v: str) -> str:
        if not v.startswith("0x") or len(v) != 42:
            raise ValueError("payee_address must be a valid EVM address")
        return v.lower()


class MicroPayResponse(BaseModel):
    transaction_id: uuid.UUID
    idempotency_key: str
    status: str  # settled_offchain (immediately) | failed
    amount_usdc: Decimal
    loop_type: str
    payment_intent_id: Optional[uuid.UUID]
    payer_available_balance_after: Decimal
    settled_at: str
    on_chain_tx_hash: Optional[str] = None
    message: str = "Payment settled off-chain. On-chain sweep occurs within 60 seconds."


# ── Top-Up ────────────────────────────────────────────────────────────────────

class TopUpRequest(BaseModel):
    deposit_tx_hash: str = Field(
        ..., description="USDC transfer tx hash on Base to Finogrid deposit address"
    )
    expected_amount_usdc: Optional[Decimal] = None


class TopUpResponse(BaseModel):
    agent_account_id: uuid.UUID
    deposit_tx_hash: str
    status: str  # pending_confirmation | credited
    message: str
