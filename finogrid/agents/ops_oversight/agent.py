"""
Ops & Oversight Agent — powered by FinGPT Sentiment + Forecaster.

Responsibilities (ALL off hot path):
- Watch flows, incidents, retries, partner failures, SLA drift
- Generate operational reports and alerts
- Summarize corridor health using sentiment + FX signals
- Surface anomalies to the ops team via Slack/email/dashboard

Runs on a schedule (e.g., every 15 minutes via Cloud Scheduler).
Does NOT release payouts or modify routing decisions.
"""
from __future__ import annotations

import structlog
from datetime import datetime, timezone, timedelta
from typing import Optional

log = structlog.get_logger()


class OpsOversightAgent:
    """
    Scheduled agent that monitors Finogrid's operational health.
    Reads from the DB and audit_events stream; writes summaries to ops dashboard.
    """

    def __init__(
        self,
        db_session_factory,
        sentiment_analyzer=None,
        corridor_forecaster=None,
        notifier=None,
    ):
        self.SessionLocal = db_session_factory
        self.sentiment = sentiment_analyzer
        self.forecaster = corridor_forecaster
        self.notifier = notifier

    async def run_health_check(self) -> dict:
        """
        Main scheduled task. Returns an operational health report.
        """
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corridors": {},
            "alerts": [],
            "partner_status": {},
        }

        async with self.SessionLocal() as db:
            # ── 1. SLA drift check ─────────────────────────────────────────────
            sla_alerts = await self._check_sla_drift(db)
            report["alerts"].extend(sla_alerts)

            # ── 2. Retry rate check ────────────────────────────────────────────
            retry_alerts = await self._check_retry_rates(db)
            report["alerts"].extend(retry_alerts)

            # ── 3. Held tasks (compliance review backlog) ──────────────────────
            held_count = await self._count_held_tasks(db)
            if held_count > 10:
                report["alerts"].append({
                    "type": "compliance_backlog",
                    "severity": "warning",
                    "message": f"{held_count} tasks held for manual compliance review",
                })

        # ── 4. Corridor risk signals (FinGPT Forecaster) ──────────────────────
        if self.forecaster:
            for corridor in ["BR", "AR", "VN", "IN", "AE", "ID", "PH", "NG"]:
                signal = self.forecaster.generate_corridor_risk_signal(
                    corridor_code=corridor,
                    news_sentiment_scores=[],
                )
                report["corridors"][corridor] = signal
                if signal["risk_level"] == "high":
                    report["alerts"].append({
                        "type": "corridor_risk",
                        "severity": "warning",
                        "corridor": corridor,
                        "message": f"Corridor {corridor} showing high risk signal",
                    })

        # ── 5. Stablecoin depeg check (FinGPT Forecaster) ─────────────────────
        if self.forecaster:
            for asset in ["USDT", "USDC"]:
                depeg = self.forecaster.get_stablecoin_depeg_risk(asset)
                report["partner_status"][f"{asset}_depeg"] = depeg
                if depeg.get("risk_level") == "high":
                    report["alerts"].append({
                        "type": "stablecoin_depeg",
                        "severity": "critical",
                        "asset": asset,
                        "depeg_bps": depeg.get("depeg_bps"),
                    })

        # ── 6. Notify if critical alerts ───────────────────────────────────────
        critical = [a for a in report["alerts"] if a.get("severity") == "critical"]
        if critical and self.notifier:
            await self.notifier.send_alert(critical)

        log.info(
            "ops_health_check_complete",
            alert_count=len(report["alerts"]),
            critical_count=len(critical),
        )
        return report

    async def _check_sla_drift(self, db) -> list[dict]:
        """Identify tasks that have exceeded their corridor SLA."""
        from sqlalchemy import select, func
        from ...database.models import PayoutTask, RoutingProfile
        from ...database.models.batch import TaskStatus

        alerts = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        result = await db.execute(
            select(
                PayoutTask.corridor_code,
                func.count(PayoutTask.id).label("overdue"),
            ).where(
                PayoutTask.status.in_([TaskStatus.EXECUTING, TaskStatus.RETRYING]),
                PayoutTask.created_at < cutoff,
            ).group_by(PayoutTask.corridor_code)
        )
        for row in result.all():
            alerts.append({
                "type": "sla_breach",
                "severity": "warning",
                "corridor": row.corridor_code,
                "overdue_tasks": row.overdue,
            })
        return alerts

    async def _check_retry_rates(self, db) -> list[dict]:
        """Flag corridors with high retry rates."""
        from sqlalchemy import select, func
        from ...database.models import PayoutTask
        from ...database.models.batch import TaskStatus

        alerts = []
        result = await db.execute(
            select(
                PayoutTask.corridor_code,
                func.avg(PayoutTask.retry_count).label("avg_retries"),
            ).where(PayoutTask.retry_count > 0)
            .group_by(PayoutTask.corridor_code)
        )
        for row in result.all():
            if row.avg_retries and float(row.avg_retries) > 1.5:
                alerts.append({
                    "type": "high_retry_rate",
                    "severity": "warning",
                    "corridor": row.corridor_code,
                    "avg_retries": round(float(row.avg_retries), 2),
                })
        return alerts

    async def _count_held_tasks(self, db) -> int:
        from sqlalchemy import select, func
        from ...database.models import PayoutTask
        from ...database.models.batch import TaskStatus

        result = await db.execute(
            select(func.count(PayoutTask.id)).where(
                PayoutTask.status == TaskStatus.HELD_FOR_REVIEW
            )
        )
        return result.scalar() or 0
