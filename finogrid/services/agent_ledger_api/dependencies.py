"""Dependency injection for Agent Ledger API."""
import hashlib
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select
import structlog

from .config import settings

log = structlog.get_logger()

engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

agent_api_key_header = APIKeyHeader(name="X-Agent-API-Key", auto_error=True)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def get_current_agent_account(
    api_key: str = Security(agent_api_key_header),
    db: AsyncSession = None,
):
    """Resolve X-Agent-API-Key to an active AgentAccount."""
    from ....database.models.agent_ledger import AgentAccount, AgentStatus

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    result = await db.execute(
        select(AgentAccount).where(
            AgentAccount.api_key_hash == key_hash,
            AgentAccount.status == AgentStatus.ACTIVE,
        )
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive agent API key",
        )
    return agent


async def assert_kya_status(agent_account, required_level: str = "basic"):
    """
    Gate: assert the agent has sufficient KYA before transacting.
    Called at the top of micropay and other outbound-payment endpoints.
    """
    from ....database.models.agent_ledger import KYAStatus

    order = {
        KYAStatus.UNVERIFIED: 0,
        KYAStatus.PENDING: 1,
        KYAStatus.BASIC: 2,
        KYAStatus.ENHANCED: 3,
    }
    required_order = order.get(KYAStatus(required_level), 2)
    actual_order = order.get(KYAStatus(agent_account.kya_status), 0)

    if actual_order < required_order:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"KYA level '{agent_account.kya_status}' insufficient. "
                f"Required: '{required_level}'. "
                "Submit KYA at POST /v1/agent-accounts/{id}/kya"
            ),
        )


def kya_daily_limit(kya_status: str) -> float:
    """Return the daily aggregate spending limit for a KYA level."""
    limits = {
        "basic": settings.kya_basic_daily_limit_usdc,
        "enhanced": settings.kya_enhanced_daily_limit_usdc,
    }
    return limits.get(kya_status, 0.0)
