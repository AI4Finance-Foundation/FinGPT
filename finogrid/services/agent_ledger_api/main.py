"""
Agent Ledger API — stablecoin micro-transaction service.

Port: 8100 (separate from v1 Ingress API on 8000)

Endpoints:
  POST /v1/agent-accounts                     Register master agent (no KYB)
  GET  /v1/agent-accounts/{id}/balance        Check balance + recent ledger
  POST /v1/agent-accounts/{id}/kya            Submit Know Your Agent
  GET  /v1/agent-accounts/{id}/kya            Check KYA status
  POST /v1/agent-accounts/{id}/wallets        Provision sub-wallet (owner sets loop_type)
  POST /v1/payment-intents                    Reserve closed-loop intent
  PATCH /v1/payment-intents/{id}              Supersede intent with audit note
  POST /v1/micropay                           Synchronous micro-payment (<300ms)
  POST /v1/agent-accounts/{id}/topup         Submit USDC deposit tx hash
  POST /v1/agent-accounts/{id}/withdraw       Trigger fiat withdrawal via v1 rails

Settlement flow (stablecoin → stablecoin → fiat):
  Top-up: USDC on Base → chain_watcher credits prefund_balance_usdc
  Micropay: DB transaction (off-chain settle) → on-chain sweep every 60s
  Withdraw: v1 Ingress API → Routing Engine → corridor adapter → Bridge off-ramp
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .config import settings
from .middleware.payment_required import PaymentRequiredMiddleware
from .routers import agent_accounts, kya, wallets, payment_intents, micropay, topup, health

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("agent_ledger_api_starting", port=settings.app_port, chain=settings.chain)
    yield
    log.info("agent_ledger_api_shutdown")


app = FastAPI(
    title="Finogrid Agent Ledger API",
    description="Stablecoin micro-transactions between AI agents — KYA, closed/open loop, x402",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(PaymentRequiredMiddleware)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(agent_accounts.router, prefix="/v1/agent-accounts", tags=["agent-accounts"])
app.include_router(kya.router, prefix="/v1/agent-accounts", tags=["kya"])
app.include_router(wallets.router, prefix="/v1/agent-accounts", tags=["wallets"])
app.include_router(payment_intents.router, prefix="/v1/payment-intents", tags=["payment-intents"])
app.include_router(micropay.router, prefix="/v1/micropay", tags=["micropay"])
app.include_router(topup.router, prefix="/v1/agent-accounts", tags=["topup"])
