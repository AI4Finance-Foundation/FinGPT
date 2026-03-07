"""
Bridge API Client — first-mile orchestration and stablecoin conversion.

Bridge handles:
- Fiat-to-USDT and fiat-to-USDC conversion
- Routing instructions to last-mile partners
- Status webhooks back to Finogrid

Finogrid never touches private keys or holds funds.
All fund movement happens inside Bridge's regulated infrastructure.

Docs: https://apidocs.bridge.xyz
"""
from __future__ import annotations

import httpx
import structlog
from typing import Optional

log = structlog.get_logger()


class BridgeError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Bridge API error {status_code}: {detail}")


class BridgeClient:
    """Async HTTP client for the Bridge orchestration API."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Api-Key": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def create_transfer(
        self,
        task_id: str,
        amount_usd: float,
        asset: str,                  # "usdt" | "usdc"
        corridor_code: str,
        recipient_wallet: Optional[str] = None,
        recipient_bank_data: Optional[dict] = None,
        delivery_mode: str = "wallet",
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """
        Initiate a transfer through Bridge.

        Bridge will:
        1. Accept fiat funding from the client's source account
        2. Convert to the specified stablecoin
        3. Route to wallet or partner off-ramp for fiat delivery
        """
        payload = {
            "amount": str(amount_usd),
            "currency": "usd",
            "destination": {
                "asset": asset.lower(),
                "network": "ethereum",  # TODO: per-corridor chain selection
            },
            "metadata": {
                "finogrid_task_id": task_id,
                "corridor": corridor_code,
                "delivery_mode": delivery_mode,
            },
        }

        if delivery_mode == "wallet" and recipient_wallet:
            payload["destination"]["address"] = recipient_wallet
        elif delivery_mode == "fiat" and recipient_bank_data:
            payload["destination"]["bank_account"] = recipient_bank_data

        headers = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        response = await self._client.post("/v0/transfers", json=payload, headers=headers)
        if response.status_code not in (200, 201, 202):
            raise BridgeError(response.status_code, response.text)

        result = response.json()
        log.info(
            "bridge_transfer_created",
            task_id=task_id,
            bridge_id=result.get("id"),
            status=result.get("status"),
        )
        return result

    async def get_transfer(self, bridge_transfer_id: str) -> dict:
        """Poll the status of a Bridge transfer."""
        response = await self._client.get(f"/v0/transfers/{bridge_transfer_id}")
        if response.status_code != 200:
            raise BridgeError(response.status_code, response.text)
        return response.json()

    async def cancel_transfer(self, bridge_transfer_id: str) -> dict:
        """Cancel a pending Bridge transfer."""
        response = await self._client.post(f"/v0/transfers/{bridge_transfer_id}/cancel")
        if response.status_code not in (200, 202):
            raise BridgeError(response.status_code, response.text)
        log.info("bridge_transfer_cancelled", bridge_id=bridge_transfer_id)
        return response.json()
