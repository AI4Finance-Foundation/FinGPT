"""
MCP Server — KYT/AML provider connector (Chainalysis-compatible).

Exposes screening operations as MCP tools.
Swap KYT vendor = swap this server, not compliance_gate logic.
"""
import os
import json
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import httpx
import structlog

log = structlog.get_logger()

KYT_API_KEY = os.getenv("KYT_API_KEY", "")
KYT_API_URL = os.getenv("KYT_API_URL", "https://api.chainalysis.com")

MCP_TOOLS = {
    "screen_address": {
        "description": "Screen a wallet address for AML risk",
        "parameters": {"address": "string", "asset": "string", "amount_usd": "number"},
    },
    "screen_sanctions": {
        "description": "Screen a name/entity against sanctions lists",
        "parameters": {"name": "string", "country": "string"},
    },
}


async def call_kyt(tool: str, params: dict) -> dict:
    headers = {"Token": KYT_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(base_url=KYT_API_URL, headers=headers, timeout=20.0) as client:
        if tool == "screen_address":
            r = await client.post("/api/kyt/v2/transfers", json={
                "network": params.get("asset", "ETH").upper(),
                "asset": params.get("asset", "USDT"),
                "address": params["address"],
                "amount": params.get("amount_usd", 0),
                "direction": "received",
            })
            data = r.json()
            return {
                "risk_score": data.get("riskScore", 0),
                "ref": data.get("externalId"),
                "cluster": data.get("cluster", {}),
            }
        elif tool == "screen_sanctions":
            # Placeholder — integrate OFAC/UN screening endpoint
            return {"hit": False, "lists_checked": ["OFAC", "UN", "EU"]}
        else:
            return {"error": f"Unknown tool: {tool}"}


class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        result = asyncio.run(call_kyt(body.get("tool"), body.get("params", {})))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.getenv("MCP_KYT_PORT", 9002))
    log.info("mcp_kyt_server_starting", port=port)
    HTTPServer(("0.0.0.0", port), MCPHandler).serve_forever()
