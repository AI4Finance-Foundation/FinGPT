"""Ops Console configuration."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpsConsoleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+asyncpg://finogrid:finogrid@localhost:5432/finogrid"
    ops_api_key: str = "ops_dev_key"       # Ops-level auth; separate from client API keys
    app_host: str = "0.0.0.0"
    app_port: int = 8200
    app_debug: bool = True
    allowed_origins: list[str] = ["*"]
