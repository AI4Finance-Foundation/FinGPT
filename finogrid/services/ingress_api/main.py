"""
Ingress API — entry point for all client payout requests.

Responsibilities:
- Authenticate and authorize the calling client
- Validate the batch schema
- Normalize payout instructions
- Publish a batch_events message to trigger downstream processing
- Return accepted status with batch ID

This service is deterministic and fast. It does NOT route or screen payouts.
"""
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .config import settings
from .dependencies import get_db, get_pubsub, verify_api_key
from .routers import batches, onboarding, webhooks, health

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("finogrid_ingress_starting", version="1.0.0")
    yield
    log.info("finogrid_ingress_shutdown")


app = FastAPI(
    title="Finogrid Ingress API",
    description="B2B stablecoin payout orchestration — batch intake",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.app_debug else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(batches.router, prefix="/v1/batches", tags=["batches"])
app.include_router(onboarding.router, prefix="/v1/onboarding", tags=["onboarding"])
app.include_router(webhooks.router, prefix="/v1/webhooks", tags=["webhooks"])
