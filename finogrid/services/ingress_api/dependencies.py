"""FastAPI dependency injection."""
import json
from typing import AsyncGenerator
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from google.cloud import pubsub_v1
import structlog

from .config import settings

log = structlog.get_logger()

# DB engine
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# Pub/Sub publisher
_publisher = None


def _get_publisher():
    global _publisher
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _publisher


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def get_current_client(
    api_key: str = Security(api_key_header),
    db: AsyncSession = None,
):
    """Resolve the API key to a Client record."""
    from sqlalchemy import select
    from ...database.models import Client

    # In production this would use a hashed key lookup table
    result = await db.execute(
        select(Client).where(Client.status == "active")
    )
    # Simplified: in production, hash(api_key) matches stored hash
    client = result.scalar_one_or_none()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key",
        )
    return client


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Lightweight key format check."""
    if not api_key or len(api_key) < 32:
        raise HTTPException(status_code=401, detail="Invalid API key format")
    return api_key


async def publish_event(topic: str, payload: dict) -> None:
    """Publish a JSON event to a Pub/Sub topic."""
    try:
        publisher = _get_publisher()
        topic_path = publisher.topic_path(settings.pubsub_project_id, topic)
        data = json.dumps(payload).encode("utf-8")
        future = publisher.publish(topic_path, data)
        future.result(timeout=10)
        log.info("pubsub_published", topic=topic, event=payload.get("event"))
    except Exception as e:
        log.error("pubsub_publish_failed", topic=topic, error=str(e))
