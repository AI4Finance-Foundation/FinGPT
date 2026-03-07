"""Batch intake router."""
import uuid
import json
import structlog
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import BatchCreateRequest, BatchCreateResponse, BatchStatusResponse
from ..dependencies import get_db, get_current_client, publish_event

log = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=BatchCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_batch(
    request: BatchCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    client=Depends(get_current_client),
):
    """
    Accept a payout batch from a business client.

    Steps:
    1. Validate schema (Pydantic handles this)
    2. Check client is ACTIVE and has corridor permissions
    3. Persist batch + tasks as DRAFT
    4. Publish batch_events/batch_created to trigger routing engine
    5. Return 202 Accepted with batch_id
    """
    log.info("batch_create_received", client_id=str(client.id), reference=request.reference)

    # Validate corridor permissions
    allowed_corridors = {p.corridor_code for p in client.corridor_permissions if p.enabled}
    for item in request.items:
        if item.corridor_code not in allowed_corridors:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Corridor {item.corridor_code} not permitted for this client",
            )

    # Persist batch
    from ....database.models import Batch, PayoutTask
    batch = Batch(
        client_id=client.id,
        reference=request.reference,
        total_amount_usd=sum(float(i.amount_usd) for i in request.items),
        task_count=len(request.items),
        metadata_=request.metadata,
    )
    db.add(batch)
    await db.flush()

    for item in request.items:
        task = PayoutTask(
            batch_id=batch.id,
            corridor_code=item.corridor_code,
            recipient_name=item.recipient_name,
            recipient_ref=item.external_ref,
            amount_usd=float(item.amount_usd),
            preferred_asset=item.preferred_asset,
            preferred_mode=item.preferred_mode,
            beneficiary_data={
                "wallet": item.recipient_wallet,
                "bank_account": item.recipient_bank_account,
                "mobile_money": item.recipient_mobile_money,
                **item.recipient_extra,
            },
        )
        db.add(task)

    await db.commit()
    await db.refresh(batch)

    # Publish event to trigger routing engine
    event = {
        "event": "batch_created",
        "batch_id": str(batch.id),
        "client_id": str(client.id),
        "corridor_codes": list({i.corridor_code for i in request.items}),
    }
    background_tasks.add_task(publish_event, "batch_events", event)

    log.info("batch_created", batch_id=str(batch.id), task_count=len(request.items))
    return BatchCreateResponse(
        batch_id=batch.id,
        reference=batch.reference,
        item_count=len(request.items),
    )


@router.get("/{batch_id}", response_model=BatchStatusResponse)
async def get_batch_status(
    batch_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    client=Depends(get_current_client),
):
    """Return the current status of a batch."""
    from sqlalchemy import select
    from ....database.models import Batch, PayoutTask

    result = await db.execute(
        select(Batch).where(Batch.id == batch_id, Batch.client_id == client.id)
    )
    batch = result.scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    task_result = await db.execute(
        select(PayoutTask).where(PayoutTask.batch_id == batch_id)
    )
    tasks = task_result.scalars().all()

    return BatchStatusResponse(
        batch_id=batch.id,
        reference=batch.reference,
        status=batch.status,
        total_amount_usd=batch.total_amount_usd,
        task_count=batch.task_count,
        completed_count=batch.completed_count,
        failed_count=batch.failed_count,
        tasks=[
            {
                "task_id": str(t.id),
                "corridor": t.corridor_code,
                "amount_usd": float(t.amount_usd),
                "status": t.status,
                "partner_tx_id": t.partner_tx_id,
                "resolved_asset": t.resolved_asset,
                "resolved_mode": t.resolved_mode,
            }
            for t in tasks
        ],
    )
