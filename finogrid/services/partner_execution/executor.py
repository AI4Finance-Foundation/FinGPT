"""
Partner Execution Service — submits tasks to Bridge and handles status.

Responsibilities:
- Receive compliance-cleared tasks from task_events
- Submit to Bridge via BridgeClient
- Capture partner_tx_id and initial status
- Publish execution_status_events
- Handle retries with exponential backoff
"""
from __future__ import annotations

import asyncio
import structlog
from typing import Optional

from .bridge_client import BridgeClient, BridgeError
from ...database.models.batch import TaskStatus

log = structlog.get_logger()


class ExecutionService:

    def __init__(self, bridge_client: BridgeClient, max_retries: int = 3):
        self.bridge = bridge_client
        self.max_retries = max_retries

    async def execute_task(self, task, db) -> bool:
        """
        Execute a single payout task. Returns True on success.
        Updates task record in-place (caller must commit).
        """
        task.status = TaskStatus.EXECUTING
        idempotency_key = f"finogrid-{task.id}"

        # Build recipient details
        wallet = task.beneficiary_data.get("wallet")
        bank_data = {
            k: v for k, v in task.beneficiary_data.items()
            if k in ("bank_account", "mobile_money", "account_number", "routing_number")
        }

        for attempt in range(self.max_retries + 1):
            try:
                result = await self.bridge.create_transfer(
                    task_id=str(task.id),
                    amount_usd=float(task.amount_usd),
                    asset=task.resolved_asset.lower(),
                    corridor_code=task.corridor_code,
                    recipient_wallet=wallet,
                    recipient_bank_data=bank_data or None,
                    delivery_mode=task.resolved_mode,
                    idempotency_key=idempotency_key,
                )
                task.partner_tx_id = result.get("id")
                task.status = TaskStatus.EXECUTING
                log.info("task_submitted_to_bridge", task_id=str(task.id), bridge_id=task.partner_tx_id)
                return True

            except BridgeError as e:
                if e.status_code in (400, 422):
                    # Unrecoverable — don't retry
                    task.status = TaskStatus.FAILED
                    task.failure_reason = f"Bridge validation error: {e.detail}"
                    log.error("bridge_validation_error", task_id=str(task.id), detail=e.detail)
                    return False

                if attempt < self.max_retries:
                    backoff = 2 ** attempt
                    log.warning(
                        "bridge_error_retrying",
                        task_id=str(task.id),
                        attempt=attempt + 1,
                        backoff=backoff,
                        error=str(e),
                    )
                    task.retry_count += 1
                    task.status = TaskStatus.RETRYING
                    await asyncio.sleep(backoff)
                else:
                    task.status = TaskStatus.FAILED
                    task.failure_reason = f"Bridge error after {self.max_retries} retries: {e.detail}"
                    log.error("bridge_max_retries_exceeded", task_id=str(task.id))
                    return False

        return False

    async def poll_task_status(self, task, db) -> Optional[str]:
        """Poll Bridge for the current status of an executing task."""
        if not task.partner_tx_id:
            return None
        try:
            result = await self.bridge.get_transfer(task.partner_tx_id)
            bridge_status = result.get("status")
            log.info("bridge_status_polled", task_id=str(task.id), bridge_status=bridge_status)

            if bridge_status == "completed":
                task.status = TaskStatus.COMPLETED
            elif bridge_status in ("failed", "cancelled"):
                task.status = TaskStatus.FAILED
                task.failure_reason = result.get("error_message", "Partner reported failure")

            return bridge_status
        except BridgeError as e:
            log.error("bridge_poll_failed", task_id=str(task.id), error=str(e))
            return None
