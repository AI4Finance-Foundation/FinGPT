"""
Ops & Controls Console API — port 8200.

Internal operations surface. Requires ops-level authentication (separate from
client API keys). All actions are append-only in the audit log.

Domains served:
  /v1/ops/search          — unified search across batches, agents, intents, transactions
  /v1/ops/exceptions      — manual review queue (held tasks, failed intents, KYA blocks)
  /v1/ops/approvals       — mandate threshold approvals (human-in-the-loop)
  /v1/ops/ledger          — ledger explorer (cross-entity view)
  /v1/ops/agents          — agent activity explorer (KYA status, wallet activity)
  /v1/ops/mandates        — mandate lifecycle console (activate / suspend / revoke)
  /v1/ops/corridors       — corridor analytics (volume, error rates, latency)
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .config import OpsConsoleSettings
from .routers import search, exceptions, approvals, ledger_explorer, agents_explorer, mandates_console, corridors

log = structlog.get_logger()
settings = OpsConsoleSettings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ops_console_starting", port=settings.app_port)
    yield
    log.info("ops_console_shutdown")


app = FastAPI(
    title="Finogrid Ops Console API",
    description="Internal operations, monitoring, and manual controls",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST", "PATCH"],
    allow_headers=["*"],
)

app.include_router(search.router,             prefix="/v1/ops/search",     tags=["search"])
app.include_router(exceptions.router,         prefix="/v1/ops/exceptions", tags=["exceptions"])
app.include_router(approvals.router,          prefix="/v1/ops/approvals",  tags=["approvals"])
app.include_router(ledger_explorer.router,    prefix="/v1/ops/ledger",     tags=["ledger"])
app.include_router(agents_explorer.router,    prefix="/v1/ops/agents",     tags=["agents"])
app.include_router(mandates_console.router,   prefix="/v1/ops/mandates",   tags=["mandates"])
app.include_router(corridors.router,          prefix="/v1/ops/corridors",  tags=["corridors"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "finogrid-ops-console"}
