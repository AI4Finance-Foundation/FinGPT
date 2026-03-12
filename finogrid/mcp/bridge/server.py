"""
MCP Server — Bridge API connector.

Exposes Bridge API operations as MCP tools so any Finogrid agent or service
can call them via Model Context Protocol without hardcoding HTTP logic.

Swapping Bridge for another partner = swap this MCP server, not core logic.
"""
import os
import json
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import httpx
import structlog

log = structlog.get_logger()

BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY", "")
BRIDGE_API_URL = os.getenv("BRIDGE_API_URL", "https://api.bridge.xyz/v0")

MCP_TOOLS = {
    "create_transfer": {
        "description": "Initiate a stablecoin transfer via Bridge",
        "parameters": {
            "amount": "string",
            "asset": "string",  # usdt | usdc
            "delivery_mode": "string",  # wallet | fiat
            "recipient_address": "string",
            "corridor": "string",
            "idempotency_key": "string",
        },
    },
    "get_transfer": {
        "description": "Get status of a Bridge transfer",
        "parameters": {"transfer_id": "string"},
    },
    "cancel_transfer": {
        "description": "Cancel a pending Bridge transfer",
        "parameters": {"transfer_id": "string"},
    },
}


async def call_bridge(tool: str, params: dict) -> dict:
    async with httpx.AsyncClient(
        base_url=BRIDGE_API_URL,
        headers={"Api-Key": BRIDGE_API_KEY},
        timeout=30.0,
    ) as client:
        if tool == "create_transfer":
            payload = {
                "amount": params["amount"],
                "currency": "usd",
                "destination": {
                    "asset": params["asset"],
                    "address": params.get("recipient_address"),
                },
                "metadata": {"corridor": params.get("corridor")},
            }
            r = await client.post("/transfers", json=payload,
                                   headers={"Idempotency-Key": params.get("idempotency_key", "")})
        elif tool == "get_transfer":
            r = await client.get(f"/transfers/{params['transfer_id']}")
        elif tool == "cancel_transfer":
            r = await client.post(f"/transfers/{params['transfer_id']}/cancel")
        else:
            return {"error": f"Unknown tool: {tool}"}

        return {"status_code": r.status_code, "body": r.json()}


class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        tool = body.get("tool")
        params = body.get("params", {})
        result = asyncio.run(call_bridge(tool, params))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, *args):
        pass  # suppress default access log


if __name__ == "__main__":
    port = int(os.getenv("MCP_BRIDGE_PORT", 9001))
    log.info("mcp_bridge_server_starting", port=port)
    HTTPServer(("0.0.0.0", port), MCPHandler).serve_forever()
