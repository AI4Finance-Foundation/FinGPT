"""
Reconciliation Service — syncs partner status and produces client-facing reports.

Responsibilities:
- Poll Bridge for status of all executing tasks
- Transition task states (executing → completed | failed)
- Update batch completion counters
- Trigger client webhooks for status changes
- Generate exportable reconciliation reports (CSV/JSON)
- Feed audit_events Pub/Sub topic for the Audit Agent
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Optional
import structlog

log = structlog.get_logger()


class ReconciliationService:

    def __init__(self, bridge_client, publisher, pubsub_project: str, db_session_factory):
        self.bridge = bridge_client
        self.publisher = publisher
        self.pubsub_project = pubsub_project
        self.SessionLocal = db_session_factory

    async def reconcile_batch(self, batch_id: str) -> dict:
        """
        Sync status for all tasks in a batch.
        Returns a summary dict.
        """
        from sqlalchemy import select
        from ...database.models import Batch, PayoutTask
        from ...database.models.batch import TaskStatus

        async with self.SessionLocal() as db:
            result = await db.execute(
                select(PayoutTask).where(
                    PayoutTask.batch_id == batch_id,
                    PayoutTask.status.in_([TaskStatus.EXECUTING, TaskStatus.RETRYING]),
                )
            )
            tasks = result.scalars().all()

            completed = 0
            failed = 0
            still_running = 0

            for task in tasks:
                bridge_status = await self._poll_task(task)
                if bridge_status == "completed":
                    completed += 1
                    task.status = TaskStatus.COMPLETED
                elif bridge_status in ("failed", "cancelled"):
                    failed += 1
                    task.status = TaskStatus.FAILED
                else:
                    still_running += 1

            # Update batch counters
            batch_result = await db.execute(select(Batch).where(Batch.id == batch_id))
            batch = batch_result.scalar_one_or_none()
            if batch:
                batch.completed_count += completed
                batch.failed_count += failed
                if still_running == 0:
                    if batch.failed_count == 0:
                        from ...database.models.batch import BatchStatus
                        batch.status = BatchStatus.COMPLETED
                    elif batch.completed_count == 0:
                        batch.status = BatchStatus.FAILED
                    else:
                        batch.status = BatchStatus.PARTIALLY_COMPLETED

            await db.commit()

            summary = {
                "batch_id": batch_id,
                "completed": completed,
                "failed": failed,
                "still_running": still_running,
                "reconciled_at": datetime.now(timezone.utc).isoformat(),
            }
            self._publish_audit_event("batch_reconciled", summary)
            return summary

    async def _poll_task(self, task) -> Optional[str]:
        if not task.partner_tx_id:
            return None
        try:
            result = await self.bridge.get_transfer(task.partner_tx_id)
            return result.get("status")
        except Exception as e:
            log.error("poll_failed", task_id=str(task.id), error=str(e))
            return None

    async def generate_report(self, batch_id: str, format: str = "json") -> str:
        """
        Generate a client-facing payout report for a batch.
        Returns JSON string or CSV string.
        """
        from sqlalchemy import select
        from ...database.models import Batch, PayoutTask

        async with self.SessionLocal() as db:
            task_result = await db.execute(
                select(PayoutTask).where(PayoutTask.batch_id == batch_id)
            )
            tasks = task_result.scalars().all()

        rows = [
            {
                "task_id": str(t.id),
                "corridor": t.corridor_code,
                "recipient": t.recipient_name,
                "amount_usd": float(t.amount_usd),
                "asset": t.resolved_asset,
                "mode": t.resolved_mode,
                "partner": t.partner_route,
                "partner_tx_id": t.partner_tx_id,
                "status": t.status,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "updated_at": t.updated_at.isoformat() if t.updated_at else None,
            }
            for t in tasks
        ]

        if format == "csv":
            buf = io.StringIO()
            if rows:
                writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            return buf.getvalue()

        return json.dumps(rows, indent=2)

    def _publish_audit_event(self, event_type: str, payload: dict):
        from google.cloud import pubsub_v1
        topic_path = self.publisher.topic_path(self.pubsub_project, "audit_events")
        data = json.dumps({"event": event_type, **payload}).encode()
        self.publisher.publish(topic_path, data)
