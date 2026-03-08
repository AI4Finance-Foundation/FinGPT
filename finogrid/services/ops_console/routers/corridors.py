"""
Corridor analytics — volume, error rates, partner latency.
Read-only. Aggregates from batches, payout_tasks, and execution_events.
"""
import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..dependencies import get_db, require_ops_key
from .....database.models.batch import PayoutTask, TaskStatus

log = structlog.get_logger()
router = APIRouter(dependencies=[Depends(require_ops_key)])


@router.get("")
async def corridor_analytics(
    since_hours: int = Query(24, description="Lookback window in hours"),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate payout_task stats per corridor for the given lookback window.
    Returns: volume, success rate, error count, hold count per corridor.
    """
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    try:
        result = await db.execute(
            select(
                PayoutTask.corridor,
                PayoutTask.status,
                func.count(PayoutTask.id).label("count"),
            )
            .where(PayoutTask.created_at >= cutoff)
            .group_by(PayoutTask.corridor, PayoutTask.status)
        )
        rows = result.all()
    except Exception as exc:
        log.warning("corridor_analytics_query_error", error=str(exc))
        return {"corridors": [], "since_hours": since_hours}

    # Reshape: corridor → {status: count}
    corridor_map: dict = {}
    for row in rows:
        corridor = row.corridor or "unknown"
        if corridor not in corridor_map:
            corridor_map[corridor] = {
                "corridor": corridor,
                "total": 0,
                "completed": 0,
                "failed": 0,
                "held": 0,
                "pending": 0,
            }
        corridor_map[corridor]["total"] += row.count
        status_key = str(row.status).lower()
        if status_key in corridor_map[corridor]:
            corridor_map[corridor][status_key] += row.count

    # Compute success rate
    for data in corridor_map.values():
        total = data["total"]
        data["success_rate_pct"] = (
            round(data["completed"] / total * 100, 1) if total > 0 else 0.0
        )
        data["error_rate_pct"] = (
            round(data["failed"] / total * 100, 1) if total > 0 else 0.0
        )

    return {
        "since_hours": since_hours,
        "corridors": sorted(corridor_map.values(), key=lambda x: x["total"], reverse=True),
    }
