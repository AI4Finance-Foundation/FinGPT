"""Webhook receiver — accepts inbound partner status callbacks."""
import structlog
from fastapi import APIRouter, Request, HTTPException, Header
from typing import Optional

log = structlog.get_logger()
router = APIRouter()


@router.post("/bridge")
async def bridge_webhook(request: Request, x_bridge_signature: Optional[str] = Header(None)):
    """Receive status updates from Bridge."""
    body = await request.json()
    log.info("bridge_webhook_received", event_type=body.get("event_type"))
    # TODO: verify HMAC signature, update execution_events, trigger reconciliation
    return {"received": True}


@router.post("/kyt")
async def kyt_webhook(request: Request):
    """Receive screening result callbacks from KYT/AML provider."""
    body = await request.json()
    log.info("kyt_webhook_received", ref=body.get("ref"))
    # TODO: update compliance_result on payout_task, trigger release or hold
    return {"received": True}
