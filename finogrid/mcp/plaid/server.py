"""
Plaid MCP Server — port 9006.

Bank account connectivity for Finogrid's Collections flow (Flow 3 from product thesis).
Abstracts Plaid Link + Plaid Transfer behind the standard MCP tool interface.

Capabilities (MVP scope):
  - Link a bank account via Plaid Link token flow
  - Verify a linked account (identity + balance)
  - Initiate an ACH pull (collection from payer bank → Finogrid USDC)
  - Retrieve transfer status
  - Webhook ingestion for Plaid transfer events

Tools:
  - create_link_token       Start a Plaid Link session for a client
  - exchange_public_token   Exchange Link public_token for access_token
  - get_account_info        Return verified account metadata + balance
  - initiate_ach_pull       Create a Plaid Transfer (ACH debit from bank)
  - get_transfer_status     Poll Plaid Transfer status
  - handle_webhook          Process Plaid transfer webhook event

Plaid docs: https://plaid.com/docs/api/
"""
import uuid
import structlog
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger()

# ── Settings ─────────────────────────────────────────────────────────────────

class PlaidSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    plaid_client_id: str = ""
    plaid_secret: str = ""
    plaid_env: str = "sandbox"          # sandbox | development | production
    plaid_country_codes: list[str] = ["US", "GB", "CA"]
    plaid_redirect_uri: str = ""
    app_host: str = "0.0.0.0"
    app_port: int = 9006

settings = PlaidSettings()

# In-memory token store for MVP (production: AlloyDB)
_access_tokens: dict[str, dict] = {}      # client_id → {access_token, item_id, account_id}
_transfers: dict[str, dict] = {}           # transfer_id → status record

# ── Pydantic models ───────────────────────────────────────────────────────────

class CreateLinkTokenRequest(BaseModel):
    client_id: str
    user_legal_name: str
    user_email: str
    products: list[str] = ["auth", "transactions"]

class ExchangePublicTokenRequest(BaseModel):
    client_id: str
    public_token: str

class GetAccountInfoRequest(BaseModel):
    client_id: str

class InitiateAchPullRequest(BaseModel):
    client_id: str
    amount_usd: float
    description: str = "Finogrid USDC top-up"
    idempotency_key: str

class GetTransferStatusRequest(BaseModel):
    transfer_id: str

# ── Plaid client helper ───────────────────────────────────────────────────────

def _get_plaid_client():
    """Return a configured plaid.ApiClient instance."""
    try:
        import plaid
        from plaid.api import plaid_api
        from plaid.model.products import Products
        env_map = {
            "sandbox": plaid.Environment.Sandbox,
            "development": plaid.Environment.Development,
            "production": plaid.Environment.Production,
        }
        configuration = plaid.Configuration(
            host=env_map.get(settings.plaid_env, plaid.Environment.Sandbox),
            api_key={
                "clientId": settings.plaid_client_id,
                "secret": settings.plaid_secret,
            },
        )
        return plaid.ApiClient(configuration), plaid_api
    except ImportError:
        return None, None

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Finogrid Plaid MCP", version="1.0.0")

@app.get("/health")
async def health():
    plaid_configured = bool(settings.plaid_client_id and settings.plaid_secret)
    return {
        "status": "ok",
        "service": "finogrid-plaid-mcp",
        "env": settings.plaid_env,
        "plaid_configured": plaid_configured,
    }


@app.post("/tools/create_link_token")
async def create_link_token(request: CreateLinkTokenRequest):
    """
    Create a Plaid Link token to initiate the bank-linking flow.
    Returns link_token — client renders Plaid Link UI with this token.
    """
    api_client, plaid_api = _get_plaid_client()

    if api_client is None or not settings.plaid_client_id:
        # Sandbox/demo mode: return mock token
        mock_token = f"link-sandbox-{uuid.uuid4().hex[:24]}"
        log.info("plaid_link_token_mock", client_id=request.client_id, token=mock_token[:20])
        return {
            "link_token": mock_token,
            "expiration": "2026-03-15T00:00:00Z",
            "request_id": str(uuid.uuid4()),
            "mode": "mock",
        }

    try:
        from plaid.model.link_token_create_request import LinkTokenCreateRequest
        from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
        from plaid.model.products import Products
        from plaid.model.country_code import CountryCode

        api = plaid_api.PlaidApi(api_client)
        plaid_request = LinkTokenCreateRequest(
            user=LinkTokenCreateRequestUser(client_user_id=request.client_id),
            client_name="Finogrid",
            products=[Products(p) for p in request.products],
            country_codes=[CountryCode(cc) for cc in settings.plaid_country_codes],
            language="en",
            redirect_uri=settings.plaid_redirect_uri or None,
        )
        response = api.link_token_create(plaid_request)
        log.info("plaid_link_token_created", client_id=request.client_id)
        return {
            "link_token": response.link_token,
            "expiration": response.expiration.isoformat(),
            "request_id": response.request_id,
        }
    except Exception as exc:
        log.error("plaid_create_link_token_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Plaid API error: {exc}")


@app.post("/tools/exchange_public_token")
async def exchange_public_token(request: ExchangePublicTokenRequest):
    """
    Exchange a Plaid public_token (from Link callback) for a permanent access_token.
    Stores the access_token server-side — client never sees it.
    """
    api_client, plaid_api = _get_plaid_client()

    if api_client is None or not settings.plaid_client_id:
        # Mock mode
        mock_access_token = f"access-sandbox-{uuid.uuid4().hex[:36]}"
        mock_item_id = f"item-{uuid.uuid4().hex[:16]}"
        _access_tokens[request.client_id] = {
            "access_token": mock_access_token,
            "item_id": mock_item_id,
        }
        log.info("plaid_token_exchange_mock", client_id=request.client_id)
        return {"item_id": mock_item_id, "linked": True, "mode": "mock"}

    try:
        from plaid.model.item_public_token_exchange_request import ItemPublicTokenExchangeRequest
        api = plaid_api.PlaidApi(api_client)
        exchange_request = ItemPublicTokenExchangeRequest(public_token=request.public_token)
        response = api.item_public_token_exchange(exchange_request)

        _access_tokens[request.client_id] = {
            "access_token": response.access_token,
            "item_id": response.item_id,
        }
        log.info("plaid_token_exchanged", client_id=request.client_id, item_id=response.item_id)
        return {"item_id": response.item_id, "linked": True}
    except Exception as exc:
        log.error("plaid_exchange_token_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Plaid API error: {exc}")


@app.post("/tools/get_account_info")
async def get_account_info(request: GetAccountInfoRequest):
    """
    Return verified bank account metadata and real-time balance.
    Used by Collections flow to verify sufficient funds before initiating ACH pull.
    """
    token_data = _access_tokens.get(request.client_id)
    if not token_data:
        raise HTTPException(status_code=404, detail="No linked bank account for this client_id")

    api_client, plaid_api = _get_plaid_client()

    if api_client is None or not settings.plaid_client_id:
        return {
            "client_id": request.client_id,
            "account_id": f"acct-mock-{request.client_id[:8]}",
            "account_name": "Mock Checking Account",
            "account_type": "depository",
            "account_subtype": "checking",
            "routing_number": "021000021",
            "available_balance_usd": 50000.00,
            "current_balance_usd": 50000.00,
            "mode": "mock",
        }

    try:
        from plaid.model.auth_get_request import AuthGetRequest
        api = plaid_api.PlaidApi(api_client)
        auth_response = api.auth_get(AuthGetRequest(access_token=token_data["access_token"]))
        account = auth_response.accounts[0]

        return {
            "client_id": request.client_id,
            "account_id": account.account_id,
            "account_name": account.name,
            "account_type": str(account.type),
            "account_subtype": str(account.subtype),
            "routing_number": auth_response.numbers.ach[0].routing if auth_response.numbers.ach else None,
            "available_balance_usd": account.balances.available,
            "current_balance_usd": account.balances.current,
        }
    except Exception as exc:
        log.error("plaid_get_account_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Plaid API error: {exc}")


@app.post("/tools/initiate_ach_pull")
async def initiate_ach_pull(request: InitiateAchPullRequest):
    """
    Initiate an ACH debit (pull) from the client's linked bank account.
    This triggers: Bank → Plaid Transfer → Finogrid USDC top-up.
    Returns transfer_id for status polling.
    """
    token_data = _access_tokens.get(request.client_id)
    if not token_data:
        raise HTTPException(status_code=404, detail="No linked bank account for this client_id")

    api_client, plaid_api = _get_plaid_client()

    if api_client is None or not settings.plaid_client_id:
        transfer_id = f"tr-mock-{uuid.uuid4().hex[:16]}"
        _transfers[transfer_id] = {
            "transfer_id": transfer_id,
            "client_id": request.client_id,
            "amount_usd": request.amount_usd,
            "status": "pending",
            "description": request.description,
            "mode": "mock",
        }
        log.info("ach_pull_mock_created", client_id=request.client_id, transfer_id=transfer_id, amount=request.amount_usd)
        return {"transfer_id": transfer_id, "status": "pending", "mode": "mock"}

    try:
        from plaid.model.transfer_create_request import TransferCreateRequest
        from plaid.model.transfer_type import TransferType
        from plaid.model.transfer_network import TransferNetwork
        from plaid.model.ach_class import ACHClass
        from plaid.model.transfer_user_in_request import TransferUserInRequest

        api = plaid_api.PlaidApi(api_client)
        account_info = _access_tokens[request.client_id]

        plaid_request = TransferCreateRequest(
            access_token=account_info["access_token"],
            account_id=account_info.get("account_id", ""),
            idempotency_key=request.idempotency_key,
            type=TransferType("debit"),
            network=TransferNetwork("ach"),
            amount=str(request.amount_usd),
            ach_class=ACHClass("ppd"),
            user=TransferUserInRequest(legal_name="Finogrid Client"),
            description=request.description[:10],  # Plaid limit: 10 chars
        )
        response = api.transfer_create(plaid_request)
        transfer = response.transfer

        _transfers[transfer.id] = {
            "transfer_id": transfer.id,
            "client_id": request.client_id,
            "amount_usd": float(transfer.amount),
            "status": str(transfer.status),
        }
        log.info("ach_pull_created", client_id=request.client_id, transfer_id=transfer.id)
        return {
            "transfer_id": transfer.id,
            "status": str(transfer.status),
            "amount_usd": float(transfer.amount),
        }
    except Exception as exc:
        log.error("ach_pull_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Plaid Transfer error: {exc}")


@app.post("/tools/get_transfer_status")
async def get_transfer_status(request: GetTransferStatusRequest):
    """Poll the status of a Plaid transfer."""
    if request.transfer_id in _transfers:
        record = _transfers[request.transfer_id]
        if record.get("mode") == "mock" and record["status"] == "pending":
            # Auto-complete mock transfers after first poll
            record["status"] = "settled"

    api_client, plaid_api = _get_plaid_client()

    if api_client is None or not settings.plaid_client_id:
        record = _transfers.get(request.transfer_id, {})
        return {
            "transfer_id": request.transfer_id,
            "status": record.get("status", "unknown"),
            "mode": "mock",
        }

    try:
        from plaid.model.transfer_get_request import TransferGetRequest
        api = plaid_api.PlaidApi(api_client)
        response = api.transfer_get(TransferGetRequest(transfer_id=request.transfer_id))
        transfer = response.transfer
        return {
            "transfer_id": transfer.id,
            "status": str(transfer.status),
            "amount_usd": float(transfer.amount),
        }
    except Exception as exc:
        log.error("get_transfer_status_error", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Plaid API error: {exc}")


@app.post("/tools/handle_webhook")
async def handle_webhook(request: Request):
    """
    Process inbound Plaid webhook events.
    Plaid sends these for transfer status changes.
    In production: verify Plaid-Verification-Id header.
    """
    body = await request.json()
    webhook_type = body.get("webhook_type")
    webhook_code = body.get("webhook_code")
    transfer_id = body.get("transfer_id")

    log.info("plaid_webhook_received", type=webhook_type, code=webhook_code, transfer_id=transfer_id)

    if webhook_type == "TRANSFER" and transfer_id:
        new_status = {
            "TRANSFER_EVENTS_UPDATE": "settled",
            "TRANSFER_EVENTS_FAILED": "failed",
            "TRANSFER_EVENTS_RETURNED": "returned",
        }.get(webhook_code, "unknown")

        if transfer_id in _transfers and new_status != "unknown":
            _transfers[transfer_id]["status"] = new_status
            log.info("transfer_status_updated_via_webhook", transfer_id=transfer_id, status=new_status)

    return {"received": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
