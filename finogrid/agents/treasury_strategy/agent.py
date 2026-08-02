"""
Treasury Strategy Agent — powered by FinGPT Robo-Advisor + Forecaster.

Future-looking only. Models what happens if Finogrid later adds:
- Inventory / prefunding positions
- Stablecoin issuance products
- Pay-in products

Provides strategic modeling for leadership and investors.
Uses AgentWeb for business intelligence and corridor analysis.
Has NO ability to move funds or modify production config.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import Optional

log = structlog.get_logger()


class TreasuryStrategyAgent:
    """
    Reads-only strategic advisory agent.
    Uses FinGPT forecaster signals and corridor analytics to model scenarios.
    """

    def __init__(self, corridor_forecaster=None, db_session_factory=None, agentweb_client=None):
        self.forecaster = corridor_forecaster
        self.SessionLocal = db_session_factory
        self.agentweb = agentweb_client  # AgentWeb client for business intelligence

    async def model_prefunding_scenario(
        self,
        corridors: list[str],
        prefund_usd: float,
        asset: str = "USDT",
    ) -> dict:
        """
        Model what prefunding $X of stablecoin would do to settlement latency
        and cost for a given set of corridors.
        Purely analytical — no execution.
        """
        if not self.forecaster:
            return {"error": "Forecaster not configured"}

        results = {}
        for corridor in corridors:
            fx = self.forecaster.get_fx_returns(corridor, weeks=4)
            depeg = self.forecaster.get_stablecoin_depeg_risk(asset)
            signal = self.forecaster.generate_corridor_risk_signal(corridor, [])

            results[corridor] = {
                "prefund_usd": prefund_usd,
                "asset": asset,
                "fx_return_4w": fx.get("weekly_return_pct"),
                "stablecoin_depeg_bps": depeg.get("depeg_bps"),
                "corridor_risk": signal.get("risk_level"),
                "modeled_benefit": (
                    "Reduced latency, eliminated conversion step"
                    if signal["risk_level"] != "high" else
                    "High corridor risk — prefunding not recommended at this time"
                ),
            }

        return {
            "scenario": "prefunding_analysis",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corridors": results,
            "note": "Strategic modeling only. No funds moved. Review with CFO before implementation.",
        }

    async def get_volume_forecast(self, corridor: str, weeks_ahead: int = 4) -> dict:
        """
        Simple volume trend projection based on historical batch data.
        """
        if not self.SessionLocal:
            return {"error": "DB not configured"}

        from sqlalchemy import select, func
        from datetime import timedelta
        from ...database.models import PayoutTask

        async with self.SessionLocal() as db:
            cutoff = datetime.now(timezone.utc) - timedelta(weeks=4)
            result = await db.execute(
                select(func.count(PayoutTask.id), func.sum(PayoutTask.amount_usd))
                .where(
                    PayoutTask.corridor_code == corridor,
                    PayoutTask.created_at >= cutoff,
                )
            )
            row = result.one()
            avg_weekly_tasks = (row[0] or 0) / 4
            avg_weekly_volume = float(row[1] or 0) / 4

        return {
            "corridor": corridor,
            "forecast_weeks": weeks_ahead,
            "avg_weekly_tasks": round(avg_weekly_tasks, 0),
            "avg_weekly_volume_usd": round(avg_weekly_volume, 2),
            "projected_volume_usd": round(avg_weekly_volume * weeks_ahead, 2),
            "note": "Linear projection from last 4 weeks. Adjust for seasonality.",
        }

    async def get_corridor_business_intelligence(
        self,
        corridor: str,
        search_query: str = None
    ) -> dict:
        """
        Get business intelligence for a corridor using AgentWeb.
        Useful for understanding local business landscape and economic conditions.
        """
        if not self.agentweb or not self.agentweb.is_configured():
            return {
                "corridor": corridor,
                "error": "AgentWeb client not configured",
                "note": "Set AGENTWEB_API_KEY in environment to enable business intelligence"
            }

        try:
            # Map corridor codes to approximate countries for search
            country_map = {
                "NG": "NG",  # Nigeria
                "BR": "BR",  # Brazil
                "AR": "AR",  # Argentina
                "IN": "IN",  # India
                "ID": "ID",  # Indonesia
                "PH": "PH",  # Philippines
                "VN": "VN",  # Vietnam
                "AE": "AE",  # UAE
                "US": "US",  # United States
            }
            
            country = country_map.get(corridor, "US")
            query = search_query or f"financial services {corridor}"
            
            # Search for relevant businesses in the corridor
            business_results = await self.agentweb.search_businesses(
                query=query,
                country=country,
                limit=10
            )
            
            if "error" in business_results:
                return {
                    "corridor": corridor,
                    "error": business_results["error"]
                }
            
            businesses = business_results.get("businesses", [])
            
            # Analyze business landscape
            categories = {}
            total_businesses = len(businesses)
            
            for business in businesses:
                category = business.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1
            
            return {
                "corridor": corridor,
                "country": country,
                "total_businesses_found": total_businesses,
                "category_distribution": categories,
                "top_businesses": businesses[:5],
                "search_query": query,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            log.error("corridor_intelligence_error", corridor=corridor, error=str(exc))
            return {
                "corridor": corridor,
                "error": str(exc)
            }

    async def enhance_corridor_analysis(
        self,
        corridor: str,
        business_context: str = None
    ) -> str:
        """
        Enhance corridor analysis with AgentWeb business data.
        Returns formatted context for strategic decision-making.
        """
        if not self.agentweb or not self.agentweb.is_configured():
            return f"Business intelligence not available for {corridor} corridor."

        try:
            intelligence = await self.get_corridor_business_intelligence(
                corridor=corridor,
                search_query=business_context
            )
            
            if "error" in intelligence:
                return f"Error retrieving business intelligence for {corridor}: {intelligence['error']}"
            
            # Format intelligence data for consumption
            context_parts = [
                f"Corridor: {corridor}",
                f"Country: {intelligence.get('country', 'N/A')}",
                f"Businesses analyzed: {intelligence.get('total_businesses_found', 0)}",
            ]
            
            # Add category breakdown
            categories = intelligence.get("category_distribution", {})
            if categories:
                top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:3]
                category_str = ", ".join([f"{cat} ({count})" for cat, count in top_categories])
                context_parts.append(f"Top categories: {category_str}")
            
            return " | ".join(context_parts)
        except Exception as exc:
            log.error("analysis_enhancement_error", corridor=corridor, error=str(exc))
            return f"Error enhancing analysis for {corridor}: {str(exc)}"
