"""Ops Console shared dependencies."""
from fastapi import Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from .config import OpsConsoleSettings

settings = OpsConsoleSettings()
engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def require_ops_key(x_ops_api_key: str = Header(...)):
    """Ops-level API key gate. Separate from client API keys."""
    if x_ops_api_key != settings.ops_api_key:
        raise HTTPException(status_code=401, detail="Invalid ops API key")
    return x_ops_api_key
