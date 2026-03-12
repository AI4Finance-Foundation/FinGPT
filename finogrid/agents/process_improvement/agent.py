"""
Process Improvement Agent — powered by FinGPT Forecaster + Sentiment.

Responsibilities (ALL off hot path):
- Analyze throughput, error patterns, country performance
- Recommend changes to routing rules or UX
- Identify underperforming corridors or partners
- Generate weekly improvement reports for product/ops team

Runs weekly via Cloud Scheduler. Outputs structured recommendations.
Does NOT modify routing config directly — all recommendations go to humans for review.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional

log = structlog.get_logger()


class ProcessImprovementAgent:
    """
    Analyzes Finogrid's performance data and produces actionable recommendations.
    """

    def __init__(
        self,
        db_session_factory,
        sentiment_analyzer=None,
        corridor_forecaster=None,
        news_fetcher=None,
    ):
        self.SessionLocal = db_session_factory
        self.sentiment = sentiment_analyzer
        self.forecaster = corridor_forecaster
        self.news_fetcher = news_fetcher

    async def generate_weekly_report(self) -> dict:
        """
        Generate a weekly process improvement report.
        Combines DB analytics, FinGPT sentiment, and FX signals.
        """
        report = {
            "period": "last_7_days",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corridor_performance": {},
            "partner_performance": {},
            "recommendations": [],
            "fingpt_signals": {},
        }

        async with self.SessionLocal() as db:
            corridor_stats = await self._get_corridor_stats(db)
            partner_stats = await self._get_partner_stats(db)

        report["corridor_performance"] = corridor_stats
        report["partner_performance"] = partner_stats

        # ── FinGPT corridor risk signals ───────────────────────────────────────
        if self.forecaster:
            for corridor in ["BR", "AR", "VN", "IN", "AE", "ID", "PH", "NG"]:
                news_scores = []
                if self.news_fetcher and self.sentiment:
                    news = await self.news_fetcher.fetch_corridor_news(corridor, days=7)
                    news_scores = self.sentiment.score_corridor_news(news, corridor)

                signal = self.forecaster.generate_corridor_risk_signal(
                    corridor_code=corridor,
                    news_sentiment_scores=news_scores,
                )
                report["fingpt_signals"][corridor] = signal

        # ── Generate recommendations ───────────────────────────────────────────
        report["recommendations"] = self._generate_recommendations(
            corridor_stats, partner_stats, report["fingpt_signals"]
        )

        log.info(
            "process_improvement_report_generated",
            recommendations=len(report["recommendations"]),
        )
        return report

    async def _get_corridor_stats(self, db) -> dict:
        from sqlalchemy import select, func
        from ...database.models import PayoutTask
        from ...database.models.batch import TaskStatus

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        result = await db.execute(
            select(
                PayoutTask.corridor_code,
                func.count(PayoutTask.id).label("total"),
                func.sum(
                    (PayoutTask.status == TaskStatus.COMPLETED).cast(db.bind.dialect.Integer)
                ).label("completed"),
                func.sum(
                    (PayoutTask.status == TaskStatus.FAILED).cast(db.bind.dialect.Integer)
                ).label("failed"),
                func.avg(PayoutTask.retry_count).label("avg_retries"),
            ).where(PayoutTask.created_at >= cutoff)
            .group_by(PayoutTask.corridor_code)
        )
        stats = {}
        for row in result.all():
            total = row.total or 1
            stats[row.corridor_code] = {
                "total": row.total,
                "completed": row.completed or 0,
                "failed": row.failed or 0,
                "success_rate": round((row.completed or 0) / total * 100, 1),
                "avg_retries": round(float(row.avg_retries or 0), 2),
            }
        return stats

    async def _get_partner_stats(self, db) -> dict:
        from sqlalchemy import select, func
        from ...database.models import PayoutTask
        from ...database.models.batch import TaskStatus

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        result = await db.execute(
            select(
                PayoutTask.partner_route,
                func.count(PayoutTask.id).label("total"),
                func.avg(PayoutTask.retry_count).label("avg_retries"),
            ).where(
                PayoutTask.created_at >= cutoff,
                PayoutTask.partner_route.isnot(None),
            ).group_by(PayoutTask.partner_route)
        )
        stats = {}
        for row in result.all():
            stats[row.partner_route] = {
                "total_tasks": row.total,
                "avg_retries": round(float(row.avg_retries or 0), 2),
            }
        return stats

    def _generate_recommendations(
        self,
        corridor_stats: dict,
        partner_stats: dict,
        fingpt_signals: dict,
    ) -> list[dict]:
        recommendations = []

        for corridor, stats in corridor_stats.items():
            # Low success rate
            if stats.get("success_rate", 100) < 90:
                recommendations.append({
                    "type": "corridor_performance",
                    "priority": "high",
                    "corridor": corridor,
                    "issue": f"Success rate {stats['success_rate']}% — below 90% threshold",
                    "action": "Review partner SLA and escalate to Bridge account manager",
                })

            # High retry rate
            if stats.get("avg_retries", 0) > 1.0:
                recommendations.append({
                    "type": "retry_rate",
                    "priority": "medium",
                    "corridor": corridor,
                    "issue": f"Average {stats['avg_retries']} retries per task",
                    "action": "Investigate partner reliability; consider enabling fallback route",
                })

            # FinGPT signal warning
            signal = fingpt_signals.get(corridor, {})
            if signal.get("risk_level") == "high":
                recommendations.append({
                    "type": "corridor_macro_risk",
                    "priority": "medium",
                    "corridor": corridor,
                    "issue": f"FinGPT signal: {signal.get('signal')} — {signal.get('avg_news_sentiment')} sentiment",
                    "action": "Consider lowering per-transaction limits or alerting clients",
                })

        return recommendations
