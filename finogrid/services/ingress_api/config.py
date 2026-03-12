from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import list


class IngressSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    database_url: str
    pubsub_project_id: str
    cors_origins: list[str] = ["*"]

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60


settings = IngressSettings()
