"""
Wallet Factory MCP Server — port 9004.

On-chain wallet registration and verification service.
Abstracts wallet provider behind MCP tool interface:
  - Coinbase Wallet SDK (Base L2)
  - Circle Programmable Wallets (future)
  - Self-custody EVM wallets (current MVP: owner provides address)

Tools:
  - register_wallet      Record wallet registration for an agent
  - verify_ownership     EIP-191 signed message ownership verification
  - get_wallet_balance   Query on-chain USDC balance for a wallet address
  - check_tx_confirmed   Verify a USDC transfer tx hash on Base
"""
import uuid
import structlog
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

log = structlog.get_logger()

# ── Settings ─────────────────────────────────────────────────────────────────

class WalletFactorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    base_rpc_url: str = "https://mainnet.base.org"
    usdc_contract_address_base: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
    chain_enabled: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 9004

settings = WalletFactorySettings()

# In-memory registry for MVP
_wallet_registry: dict[str, dict] = {}

USDC_DECIMALS = 6
USDC_TRANSFER_ABI = [
    {
        "inputs": [
            {"name": "account", "type": "address"}
        ],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    },
]

# ── Pydantic models ───────────────────────────────────────────────────────────

class RegisterWalletRequest(BaseModel):
    wallet_address: str
    agent_account_id: str
    chain: str = "base"
    loop_type: str = "open"


class VerifyOwnershipRequest(BaseModel):
    wallet_address: str
    message: str
    signature: str


class GetWalletBalanceRequest(BaseModel):
    wallet_address: str
    chain: str = "base"


class CheckTxConfirmedRequest(BaseModel):
    tx_hash: str
    deposit_address: str
    usdc_contract: str
    chain: str = "base"

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Finogrid Wallet Factory MCP", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "finogrid-wallet-factory", "chain_enabled": settings.chain_enabled}


@app.post("/tools/register_wallet")
async def register_wallet(request: RegisterWalletRequest):
    """
    Register an EVM wallet address for an agent.
    Validates address format and records in registry.
    Does NOT require chain connectivity for MVP.
    """
    addr = request.wallet_address.lower()
    if not addr.startswith("0x") or len(addr) != 42:
        raise HTTPException(status_code=400, detail="Invalid EVM address format")

    registry_key = f"{request.agent_account_id}:{addr}"
    if registry_key in _wallet_registry:
        return {
            "registered": True,
            "wallet_address": addr,
            "message": "Wallet already registered",
        }

    _wallet_registry[registry_key] = {
        "wallet_address": addr,
        "agent_account_id": request.agent_account_id,
        "chain": request.chain,
        "loop_type": request.loop_type,
        "registered_at": str(uuid.uuid4()),  # placeholder timestamp
    }

    log.info("wallet_registered", wallet=addr, agent_id=request.agent_account_id, chain=request.chain)
    return {
        "registered": True,
        "wallet_address": addr,
        "chain": request.chain,
        "message": "Wallet registered successfully",
    }


@app.post("/tools/verify_ownership")
async def verify_ownership(request: VerifyOwnershipRequest):
    """
    Verify EIP-191 signed message to prove wallet ownership.
    The agent owner must sign the message with their wallet's private key.
    """
    if not settings.chain_enabled:
        return {
            "verified": True,
            "wallet_address": request.wallet_address.lower(),
            "message": "Ownership verification skipped (chain_enabled=False)",
        }

    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct

        msg = encode_defunct(text=request.message)
        recovered = Account.recover_message(msg, signature=request.signature)

        verified = recovered.lower() == request.wallet_address.lower()
        return {
            "verified": verified,
            "wallet_address": request.wallet_address.lower(),
            "recovered_address": recovered.lower(),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("ownership_verification_failed", error=str(exc))
        return {"verified": False, "error": str(exc)}


@app.post("/tools/get_wallet_balance")
async def get_wallet_balance(request: GetWalletBalanceRequest):
    """Query on-chain USDC balance for a wallet address on Base."""
    if not settings.chain_enabled:
        return {
            "wallet_address": request.wallet_address.lower(),
            "usdc_balance": "0.00",
            "chain": request.chain,
            "message": "On-chain query skipped (chain_enabled=False)",
        }

    try:
        from web3 import Web3
        from decimal import Decimal

        w3 = Web3(Web3.HTTPProvider(settings.base_rpc_url))
        usdc = w3.eth.contract(
            address=w3.to_checksum_address(settings.usdc_contract_address_base),
            abi=USDC_TRANSFER_ABI,
        )
        balance_raw = usdc.functions.balanceOf(
            w3.to_checksum_address(request.wallet_address)
        ).call()
        balance = Decimal(str(balance_raw)) / Decimal(str(10 ** USDC_DECIMALS))

        return {
            "wallet_address": request.wallet_address.lower(),
            "usdc_balance": str(balance),
            "chain": request.chain,
        }
    except Exception as exc:  # noqa: BLE001
        log.error("get_wallet_balance_error", wallet=request.wallet_address, error=str(exc))
        raise HTTPException(status_code=503, detail=f"Chain query failed: {exc}")


@app.post("/tools/check_tx_confirmed")
async def check_tx_confirmed(request: CheckTxConfirmedRequest):
    """
    Verify a USDC transfer tx hash:
      - tx exists and was mined
      - recipient matches deposit_address
      - token is USDC (matches usdc_contract)
    Returns confirmed=True + amount_usdc if valid.
    """
    if not settings.chain_enabled:
        return {
            "confirmed": False,
            "tx_hash": request.tx_hash,
            "message": "On-chain verification skipped (chain_enabled=False)",
        }

    try:
        from web3 import Web3
        from decimal import Decimal

        w3 = Web3(Web3.HTTPProvider(settings.base_rpc_url))
        receipt = w3.eth.get_transaction_receipt(request.tx_hash)

        if receipt is None or receipt["status"] != 1:
            return {"confirmed": False, "tx_hash": request.tx_hash, "reason": "not_mined_or_failed"}

        usdc = w3.eth.contract(
            address=w3.to_checksum_address(request.usdc_contract),
            abi=USDC_TRANSFER_ABI,
        )
        deposit_addr = w3.to_checksum_address(request.deposit_address)

        # Parse Transfer events from receipt
        transfer_events = usdc.events.Transfer().process_receipt(receipt)
        amount_raw = 0
        for event in transfer_events:
            if event["args"]["to"].lower() == deposit_addr.lower():
                amount_raw += event["args"]["value"]

        if amount_raw == 0:
            return {"confirmed": False, "reason": "no_usdc_transfer_to_deposit_address"}

        amount_usdc = str(Decimal(str(amount_raw)) / Decimal(str(10 ** USDC_DECIMALS)))
        log.info("tx_confirmed", tx_hash=request.tx_hash, amount_usdc=amount_usdc)
        return {
            "confirmed": True,
            "tx_hash": request.tx_hash,
            "amount_usdc": amount_usdc,
            "block_number": receipt["blockNumber"],
        }

    except Exception as exc:  # noqa: BLE001
        log.error("check_tx_confirmed_error", tx_hash=request.tx_hash, error=str(exc))
        raise HTTPException(status_code=503, detail=f"Chain query failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
