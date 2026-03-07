"""Client onboarding router."""
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import OnboardingRequest, OnboardingResponse
from ..dependencies import get_db

log = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=OnboardingResponse, status_code=status.HTTP_201_CREATED)
async def register_client(
    request: OnboardingRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Register a new B2B client.

    Steps:
    1. Create client record in PENDING_KYB status
    2. Kick off KYB workflow via onboarding service
    3. Set up corridor permissions based on requested_corridors
    """
    from sqlalchemy import select
    from ....database.models import Client, ClientCorridorPermission

    # Prevent duplicate registration
    existing = await db.execute(select(Client).where(Client.email == request.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Client with this email already exists")

    client = Client(
        name=request.name,
        legal_name=request.legal_name,
        registration_country=request.registration_country.upper(),
        email=request.email,
        webhook_url=request.webhook_url,
    )
    db.add(client)
    await db.flush()

    # Add requested corridor permissions (default: disabled until KYB approved)
    supported = ["BR", "AR", "VN", "IN", "AE", "ID", "PH", "NG"]
    for corridor in request.requested_corridors:
        if corridor.upper() in supported:
            db.add(ClientCorridorPermission(
                client_id=client.id,
                corridor_code=corridor.upper(),
                enabled=False,  # Enabled after KYB approval
            ))

    await db.commit()
    log.info("client_registered", client_id=str(client.id), email=client.email)

    return OnboardingResponse(
        client_id=client.id,
        status=client.status,
        kyb_status=client.kyb_status,
        message="Client registered. KYB verification will begin shortly.",
    )
