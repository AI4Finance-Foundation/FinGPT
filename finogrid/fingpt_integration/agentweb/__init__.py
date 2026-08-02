"""
AgentWeb integration for FinGPT.

Provides business data access via AgentWeb API:
- 85M+ businesses across 195 countries
- Business search and verification
- Local search capabilities
- Contact information retrieval
"""
from .client import AgentWebClient, AGENTWEB_API_KEY, AGENTWEB_MCP_URL

__all__ = ["AgentWebClient", "AGENTWEB_API_KEY", "AGENTWEB_MCP_URL"]
