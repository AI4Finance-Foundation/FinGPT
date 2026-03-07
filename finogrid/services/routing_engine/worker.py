"""
Routing Engine Worker — subscribes to batch_events, runs routing, publishes task_events.
"""
import json
import asyncio
import structlog
from google.cloud import pubsub_v1
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select

from .engine import RoutingEngine, RoutingError
from ...database.models import Batch, PayoutTask, RoutingProfile, ComplianceProfile
from ...database.models.batch import TaskStatus

log = structlog.get_logger()


class RoutingWorker:
    def __init__(self, settings):
        self.settings = settings
        self.engine = create_async_engine(settings.database_url)
        self.SessionLocal = async_sessionmaker(self.engine, expire_on_commit=False)
        self.subscriber = pubsub_v1.SubscriberClient()
        self.publisher = pubsub_v1.PublisherClient()

    async def process_batch(self, batch_id: str):
        async with self.SessionLocal() as db:
            # Load all pending tasks for this batch
            result = await db.execute(
                select(PayoutTask).where(
                    PayoutTask.batch_id == batch_id,
                    PayoutTask.status == TaskStatus.PENDING,
                )
            )
            tasks = result.scalars().all()

            for task in tasks:
                await self._route_task(db, task)

            await db.commit()

    async def _route_task(self, db: AsyncSession, task: PayoutTask):
        task.status = TaskStatus.ROUTING

        # Load corridor profiles
        rp_result = await db.execute(
            select(RoutingProfile).where(RoutingProfile.corridor_code == task.corridor_code)
        )
        cp_result = await db.execute(
            select(ComplianceProfile).where(ComplianceProfile.corridor_code == task.corridor_code)
        )
        rp = rp_result.scalar_one_or_none()
        cp = cp_result.scalar_one_or_none()

        if not rp or not cp:
            task.status = TaskStatus.FAILED
            task.failure_reason = f"No routing/compliance profile for corridor {task.corridor_code}"
            log.error("routing_profile_missing", corridor=task.corridor_code, task_id=str(task.id))
            return

        routing = RoutingEngine(rp, cp)
        try:
            decision = routing.decide(
                task_id=str(task.id),
                corridor_code=task.corridor_code,
                amount_usd=float(task.amount_usd),
                preferred_asset=task.preferred_asset,
                preferred_mode=task.preferred_mode,
                beneficiary_data=task.beneficiary_data,
            )
            task.resolved_asset = decision.resolved_asset
            task.resolved_mode = decision.resolved_mode
            task.partner_route = decision.primary_partner
            task.fallback_route = decision.fallback_partner
            task.status = TaskStatus.COMPLIANCE_CHECK

            log.info(
                "task_routed",
                task_id=str(task.id),
                asset=decision.resolved_asset,
                mode=decision.resolved_mode,
                partner=decision.primary_partner,
            )
            # Publish to compliance gate
            self._publish("task_events", {
                "event": "task_routed",
                "task_id": str(task.id),
                "corridor": task.corridor_code,
                "asset": decision.resolved_asset,
                "mode": decision.resolved_mode,
                "partner": decision.primary_partner,
                "compliance_profile_id": decision.compliance_profile_id,
            })

        except RoutingError as e:
            task.status = TaskStatus.FAILED
            task.failure_reason = str(e)
            log.warning("routing_failed", task_id=str(task.id), reason=str(e))

    def _publish(self, topic: str, payload: dict):
        topic_path = self.publisher.topic_path(self.settings.pubsub_project_id, topic)
        self.publisher.publish(topic_path, json.dumps(payload).encode())

    def start(self, subscription: str):
        sub_path = self.subscriber.subscription_path(
            self.settings.pubsub_project_id, subscription
        )

        def callback(message):
            data = json.loads(message.data.decode())
            if data.get("event") == "batch_created":
                asyncio.get_event_loop().run_until_complete(
                    self.process_batch(data["batch_id"])
                )
            message.ack()

        future = self.subscriber.subscribe(sub_path, callback=callback)
        log.info("routing_worker_started", subscription=subscription)
        try:
            future.result()
        except KeyboardInterrupt:
            future.cancel()
