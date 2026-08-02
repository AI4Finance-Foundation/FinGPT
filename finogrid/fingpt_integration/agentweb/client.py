"""
AgentWeb Client — Business data integration for FinGPT agents.

Provides access to 85M+ businesses across 195 countries via AgentWeb API.
Key capabilities: business search, local search, company verification.

AgentWeb: https://agentweb.live
API Docs: https://api.agentweb.live/docs
"""
import os
import httpx
import structlog
from typing import Optional, Dict, Any

log = structlog.get_logger()

AGENTWEB_API_KEY = os.getenv("AGENTWEB_API_KEY", "")
AGENTWEB_MCP_URL = os.getenv("AGENTWEB_MCP_URL", "https://api.agentweb.live/mcp")


class AgentWebClient:
    """Client for AgentWeb business data API via MCP interface."""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or AGENTWEB_API_KEY
        self.base_url = base_url or AGENTWEB_MCP_URL
        
        if not self.api_key:
            log.warning("agentweb_no_api_key", message="AgentWeb API key not configured")
    
    async def _call_mcp_tool(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call AgentWeb MCP tool with proper error handling."""
        if not self.api_key:
            return {"error": "AgentWeb API key not configured"}
        
        try:
            async with httpx.AsyncClient(
                base_url=self.base_url,
                headers={"X-API-Key": self.api_key},
                timeout=30.0
            ) as client:
                payload = {
                    "tool": tool,
                    "params": params
                }
                response = await client.post("/", json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            log.error("agentweb_http_error", tool=tool, status_code=exc.response.status_code)
            return {"error": f"HTTP {exc.response.status_code}: {exc.response.text}"}
        except httpx.TimeoutException:
            log.error("agentweb_timeout", tool=tool)
            return {"error": "Request timeout"}
        except Exception as exc:
            log.error("agentweb_call_error", tool=tool, error=str(exc))
            return {"error": str(exc)}
    
    async def search_businesses(
        self, 
        query: str, 
        country: str = "US",
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search for businesses by name or query.
        
        Args:
            query: Business name or search query
            country: ISO country code (default: US)
            limit: Maximum number of results (default: 5)
        
        Returns:
            Dict with business search results or error
        """
        return await self._call_mcp_tool("search_businesses", {
            "query": query,
            "country": country,
            "limit": limit
        })
    
    async def get_business_details(self, business_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific business.
        
        Args:
            business_id: AgentWeb business identifier
        
        Returns:
            Dict with detailed business information or error
        """
        return await self._call_mcp_tool("get_business", {
            "business_id": business_id
        })
    
    async def local_search(
        self,
        query: str,
        location: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Local search for businesses and services.
        
        Args:
            query: Search query (e.g., "coffee shops")
            location: Location string (e.g., "Shoreditch, London")
            limit: Maximum number of results (default: 5)
        
        Returns:
            Dict with local search results or error
        """
        return await self._call_mcp_tool("local_search", {
            "query": query,
            "location": location,
            "limit": limit
        })
    
    async def verify_business_entity(
        self,
        business_name: str,
        country: str = "US",
        require_address: bool = True
    ) -> Dict[str, Any]:
        """
        Verify business entity existence and get basic information.
        
        Args:
            business_name: Name of the business to verify
            country: ISO country code (default: US)
            require_address: Whether to require address for verification
        
        Returns:
            Dict with verification status and business details
        """
        results = await self.search_businesses(
            query=business_name,
            country=country,
            limit=3
        )
        
        if "error" in results:
            return {
                "verified": False,
                "business_name": business_name,
                "error": results["error"]
            }
        
        businesses = results.get("businesses", [])
        
        if not businesses:
            return {
                "verified": False,
                "business_name": business_name,
                "message": "No matching businesses found"
            }
        
        # Check for exact or close match
        verified_business = None
        for business in businesses:
            name_match = business_name.lower() in business.get("name", "").lower()
            has_address = bool(business.get("address"))
            
            if name_match and (not require_address or has_address):
                verified_business = business
                break
        
        if verified_business:
            return {
                "verified": True,
                "business_name": business_name,
                "business_data": verified_business,
                "match_quality": "exact" if verified_business["name"].lower() == business_name.lower() else "partial"
            }
        
        return {
            "verified": False,
            "business_name": business_name,
            "candidates": businesses,
            "message": "No exact match found, similar businesses available"
        }
    
    async def get_business_contacts(self, business_id: str) -> Dict[str, Any]:
        """
        Get contact information for a business.
        
        Args:
            business_id: AgentWeb business identifier
        
        Returns:
            Dict with contact information or error
        """
        business_details = await self.get_business_details(business_id)
        
        if "error" in business_details:
            return business_details
        
        business = business_details.get("business", {})
        return {
            "business_id": business_id,
            "phone": business.get("phone"),
            "website": business.get("website"),
            "email": business.get("email"),
            "address": business.get("address"),
            "location": business.get("location")
        }
    
    def is_configured(self) -> bool:
        """Check if AgentWeb client is properly configured with API key."""
        return bool(self.api_key)