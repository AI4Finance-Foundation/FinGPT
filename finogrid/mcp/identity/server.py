"""
MCP Server — Identity/KYB provider connector.

Handles business onboarding verification.
Plaid used only where identity/account verification helps — not as the primary AML engine.
"""
import os
import json
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import httpx
import structlog

log = structlog.get_logger()

KYB_API_KEY = os.getenv("KYB_API_KEY", "")
KYB_API_URL = os.getenv("KYB_API_URL", "")

MCP_TOOLS = {
    "submit_kyb": {
        "description": "Submit a business for KYB verification",
        "parameters": {"legal_name": "string", "country": "string", "registration_number": "string"},
    },
    "get_kyb_status": {
        "description": "Check the status of a KYB submission",
        "parameters": {"kyb_ref": "string"},
    },
}


async def call_identity(tool: str, params: dict) -> dict:
    headers = {"Authorization": f"Bearer {KYB_API_KEY}"}
    async with httpx.AsyncClient(base_url=KYB_API_URL, headers=headers, timeout=20.0) as client:
        if tool == "submit_kyb":
            r = await client.post("/kyb/submit", json=params)
            return r.json()
        elif tool == "get_kyb_status":
            r = await client.get(f"/kyb/{params['kyb_ref']}/status")
            return r.json()
        else:
            return {"error": f"Unknown tool: {tool}"}


class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        result = asyncio.run(call_identity(body.get("tool"), body.get("params", {})))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    port = int(os.getenv("MCP_IDENTITY_PORT", 9003))
    log.info("mcp_identity_server_starting", port=port)
    HTTPServer(("0.0.0.0", port), MCPHandler).serve_forever()
