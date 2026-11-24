"""Subscription and Plan Management Endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.models.organization import Organization
from backend.app.db.models.user import User
from backend.app.schemas.subscription import (
    PlansListResponse,
    SubscriptionChangeRequest,
    SubscriptionChangeResponse
)
from backend.app.services.subscription_service import SubscriptionService
from backend.app.api.deps import require_jwt_user, require_role, get_subscription_service

router = APIRouter()

@router.get("/plans", response_model=PlansListResponse)
def list_available_plans(
    sub_service: SubscriptionService = Depends(get_subscription_service)
):
    """Lists all available commercial and evaluation SaaS subscription tiers."""
    return sub_service.get_plans()

@router.get("/current")
def get_current_subscription(
    user: User = Depends(require_jwt_user),
    db: Session = Depends(get_db)
):
    """Returns the current subscription tier, quota limits, and billing status of the caller's organization."""
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return {
        "organization_id": org.id,
        "subscription_tier": org.subscription_tier,
        "subscription_status": org.subscription_status,
        "is_commercial_cleared": org.is_commercial_cleared,
        "research_access_permitted": org.research_access_permitted,
        "monthly_request_quota": org.monthly_request_quota,
        "current_month_requests": org.current_month_requests,
        "rate_limit_per_minute": org.rate_limit_per_minute,
        "quota_reset_date": org.quota_reset_date,
        "billing_email": org.billing_email
    }

@router.post("/change-plan", response_model=SubscriptionChangeResponse)
def change_subscription_plan(
    req: SubscriptionChangeRequest,
    user: User = Depends(require_role(["OWNER", "ADMIN"])),
    db: Session = Depends(get_db),
    sub_service: SubscriptionService = Depends(get_subscription_service)
):
    """Changes subscription tier using payment provider abstraction."""
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

    try:
        res = sub_service.change_subscription(org, req.new_tier, db)
        return SubscriptionChangeResponse(**res)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Subscription update failed: {str(e)}")

@router.post("/cancel")
def cancel_subscription(
    user: User = Depends(require_role(["OWNER"])),
    db: Session = Depends(get_db),
    sub_service: SubscriptionService = Depends(get_subscription_service)
):
    """Cancels active subscription, reverting to FREE tier at end of cycle."""
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")

    org.subscription_status = "CANCELED"
    db.commit()
    return {"message": "Subscription successfully canceled.", "organization_id": org.id}
