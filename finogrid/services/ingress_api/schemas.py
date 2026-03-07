"""Pydantic schemas for the Ingress API."""
import uuid
from decimal import Decimal
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class PayoutItem(BaseModel):
    """A single payout within a batch."""
    external_ref: str = Field(..., description="Client's own reference ID for this payout")
    corridor_code: str = Field(..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2")
    recipient_name: str = Field(..., min_length=1, max_length=512)
    amount_usd: Decimal = Field(..., gt=0, le=100_000)
    preferred_asset: Optional[Literal["USDT", "USDC"]] = None
    preferred_mode: Optional[Literal["wallet", "fiat"]] = None

    # Recipient delivery details (only one of these should be populated)
    recipient_wallet: Optional[str] = Field(None, description="On-chain wallet address")
    recipient_bank_account: Optional[str] = None
    recipient_mobile_money: Optional[str] = None
    recipient_extra: dict = Field(default_factory=dict)

    @field_validator("corridor_code")
    @classmethod
    def corridor_must_be_supported(cls, v: str) -> str:
        supported = {"BR", "AR", "VN", "IN", "AE", "ID", "PH", "NG"}
        if v.upper() not in supported:
            raise ValueError(f"Corridor {v} is not supported in v1. Supported: {supported}")
        return v.upper()


class BatchCreateRequest(BaseModel):
    """Request body for creating a payout batch."""
    reference: str = Field(..., min_length=1, max_length=255, description="Client batch reference")
    items: list[PayoutItem] = Field(..., min_length=1, max_length=1000)
    metadata: dict = Field(default_factory=dict)


class BatchCreateResponse(BaseModel):
    """Accepted response for a batch creation request."""
    batch_id: uuid.UUID
    reference: str
    item_count: int
    status: str = "draft"
    message: str = "Batch accepted. Processing will begin shortly."


class BatchStatusResponse(BaseModel):
    """Current status of a batch."""
    batch_id: uuid.UUID
    reference: str
    status: str
    total_amount_usd: Decimal
    task_count: int
    completed_count: int
    failed_count: int
    tasks: list[dict] = []


class OnboardingRequest(BaseModel):
    """Business client registration."""
    name: str = Field(..., min_length=1, max_length=255)
    legal_name: str = Field(..., min_length=1, max_length=512)
    registration_country: str = Field(..., min_length=2, max_length=2)
    email: str
    webhook_url: Optional[str] = None
    requested_corridors: list[str] = Field(default_factory=list)


class OnboardingResponse(BaseModel):
    client_id: uuid.UUID
    status: str
    kyb_status: str
    message: str
