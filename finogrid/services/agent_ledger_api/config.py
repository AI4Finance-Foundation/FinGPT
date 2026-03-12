from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class AgentLedgerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8100
    log_level: str = "INFO"

    database_url: str = "postgresql+asyncpg://finogrid:password@localhost:5432/finogrid"

    # Chain (Base L2 default — aligns with x402 standard)
    chain: str = "base"
    base_rpc_url: str = "https://mainnet.base.org"
    chain_enabled: bool = False  # Set True in prod; False skips on-chain calls

    # Finogrid deposit address on Base (USDC top-ups arrive here)
    agent_ledger_deposit_address: str = "0x0000000000000000000000000000000000000000"

    # Sweep wallet (private key in GCP Secret Manager — never in DB)
    sweep_wallet_address: str = "0x0000000000000000000000000000000000000000"

    # KYA thresholds (USDC/day)
    kya_basic_daily_limit_usdc: float = 1.00
    kya_enhanced_daily_limit_usdc: float = 100.00
    kya_enabled: bool = True  # Set False in test environments

    # KYA validator MCP server
    kya_validator_mcp_url: str = "http://localhost:9005"

    # Wallet factory MCP server
    wallet_factory_mcp_url: str = "http://localhost:9004"

    # v1 Ingress API (for withdrawal routing)
    v1_ingress_url: str = "http://localhost:8000"
    v1_internal_api_key: str = "internal-service-key"

    # x402 settings
    x402_payment_protected_paths: list[str] = []  # Paths requiring payment proof
    x402_nonce_ttl_seconds: int = 300

    # Intent sweeper
    intent_sweeper_interval_seconds: int = 300

    # Chain watcher
    chain_watcher_sweep_interval_seconds: int = 60
    usdc_contract_address_base: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"


settings = AgentLedgerSettings()
