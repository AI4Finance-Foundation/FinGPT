"""
x402 Payment Required middleware.

Implements the HTTP 402 / x402 protocol:
  - Coinbase/Cloudflare standard for machine-readable payment walls
  - Inspects PAYMENT-SIGNATURE header on protected paths
  - Returns 402 with PAYMENT-REQUIRED header describing the requirement
  - On valid payment proof: forwards request to the handler

Header flow:
  Client → GET /protected/resource
  Server ← 402 Payment Required
         ← PAYMENT-REQUIRED: <base64-encoded JSON requirement>

  Client → GET /protected/resource
         → PAYMENT-SIGNATURE: <base64-encoded signed payment receipt>
  Server ← 200 OK (if valid) or 402 (if invalid/expired)
         ← PAYMENT-RESPONSE: <base64-encoded settlement receipt>

Reference: https://x402.org (Coinbase / Cloudflare)
"""
import base64
import json
import time
import hashlib
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import settings

log = structlog.get_logger()

PAYMENT_REQUIRED_HEADER = "PAYMENT-REQUIRED"
PAYMENT_SIGNATURE_HEADER = "PAYMENT-SIGNATURE"
PAYMENT_RESPONSE_HEADER = "PAYMENT-RESPONSE"


def _encode_requirement(path: str, amount_usdc: float = 0.001) -> str:
    """Encode payment requirement as base64 JSON."""
    requirement = {
        "scheme": "x402",
        "version": "1",
        "network": "base-mainnet",
        "asset": "USDC",
        "payTo": settings.agent_ledger_deposit_address,
        "maxAmountRequired": str(amount_usdc),
        "resource": path,
        "description": f"Access payment for {path}",
        "mimeType": "application/json",
        "outputSchema": None,
        "extra": {
            "name": "Finogrid Agent Ledger",
            "version": "1.0",
        },
    }
    return base64.b64encode(json.dumps(requirement).encode()).decode()


def _decode_signature(signature_b64: str) -> dict | None:
    """Decode and parse PAYMENT-SIGNATURE header."""
    try:
        decoded = base64.b64decode(signature_b64).decode()
        return json.loads(decoded)
    except Exception:  # noqa: BLE001
        return None


def _validate_payment_signature(sig_data: dict, path: str) -> tuple[bool, str]:
    """
    Validate a payment signature for x402.
    In production this verifies the on-chain USDC transfer receipt.
    For MVP: validates structure and nonce freshness.
    """
    required_fields = {"network", "asset", "payTo", "amount", "nonce", "timestamp", "resource"}
    if not required_fields.issubset(sig_data.keys()):
        return False, "Missing required signature fields"

    # Check nonce freshness (TTL from config)
    try:
        ts = float(sig_data["timestamp"])
        age = time.time() - ts
        if age > settings.x402_nonce_ttl_seconds:
            return False, f"Payment signature expired ({age:.0f}s ago, TTL={settings.x402_nonce_ttl_seconds}s)"
    except (ValueError, TypeError):
        return False, "Invalid timestamp in signature"

    # Check deposit address
    if sig_data.get("payTo", "").lower() != settings.agent_ledger_deposit_address.lower():
        return False, "payTo address mismatch"

    # Check resource matches path
    if sig_data.get("resource") != path:
        return False, f"Resource mismatch: expected '{path}', got '{sig_data.get('resource')}'"

    return True, "valid"


class PaymentRequiredMiddleware(BaseHTTPMiddleware):
    """
    Intercepts requests to x402-protected paths.
    If the path is in settings.x402_payment_protected_paths:
      - No PAYMENT-SIGNATURE: return 402 with requirement
      - Invalid PAYMENT-SIGNATURE: return 402 with error
      - Valid PAYMENT-SIGNATURE: forward to handler, attach PAYMENT-RESPONSE
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Check if this path requires payment
        protected = any(
            path.startswith(p) for p in settings.x402_payment_protected_paths
        )
        if not protected:
            return await call_next(request)

        signature_header = request.headers.get(PAYMENT_SIGNATURE_HEADER)

        if not signature_header:
            # No payment signature — return 402 with requirement
            requirement_b64 = _encode_requirement(path)
            log.info("x402_payment_required", path=path, client=str(request.client))
            return JSONResponse(
                status_code=402,
                content={
                    "error": "Payment Required",
                    "x402_version": "1",
                    "accepts": [
                        {
                            "scheme": "x402",
                            "network": "base-mainnet",
                            "asset": "USDC",
                            "payTo": settings.agent_ledger_deposit_address,
                        }
                    ],
                },
                headers={PAYMENT_REQUIRED_HEADER: requirement_b64},
            )

        # Validate signature
        sig_data = _decode_signature(signature_header)
        if sig_data is None:
            return JSONResponse(
                status_code=402,
                content={"error": "Invalid payment signature encoding"},
                headers={PAYMENT_REQUIRED_HEADER: _encode_requirement(path)},
            )

        valid, reason = _validate_payment_signature(sig_data, path)
        if not valid:
            log.warning("x402_invalid_signature", path=path, reason=reason)
            return JSONResponse(
                status_code=402,
                content={"error": f"Payment signature invalid: {reason}"},
                headers={PAYMENT_REQUIRED_HEADER: _encode_requirement(path)},
            )

        # Valid payment — forward and attach receipt
        response = await call_next(request)

        receipt = {
            "scheme": "x402",
            "network": sig_data.get("network"),
            "asset": sig_data.get("asset"),
            "amount": sig_data.get("amount"),
            "paidAt": time.time(),
            "resource": path,
            "nonce": sig_data.get("nonce"),
            "receiptId": hashlib.sha256(
                f"{sig_data.get('nonce')}{sig_data.get('timestamp')}".encode()
            ).hexdigest()[:16],
        }
        receipt_b64 = base64.b64encode(json.dumps(receipt).encode()).decode()
        response.headers[PAYMENT_RESPONSE_HEADER] = receipt_b64

        log.info("x402_payment_accepted", path=path, amount=sig_data.get("amount"))
        return response
