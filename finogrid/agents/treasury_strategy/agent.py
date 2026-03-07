"""
Treasury Strategy Agent — powered by FinGPT Robo-Advisor + Forecaster.

Future-looking only. Models what happens if Finogrid later adds:
- Inventory / prefunding positions
- Stablecoin issuance products
- Pay-in products

Provides strategic modeling for leadership and investors.
Has NO ability to move funds or modify production config.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone

log = structlog.get_logger()


class TreasuryStrategyAgent:
    """
    Reads-only strategic advisory agent.
    Uses FinGPT forecaster signals and corridor analytics to model scenarios.
    """

    def __init__(self, corridor_forecaster=None, db_session_factory=None):
        self.forecaster = corridor_forecaster
        self.SessionLocal = db_session_factory

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
