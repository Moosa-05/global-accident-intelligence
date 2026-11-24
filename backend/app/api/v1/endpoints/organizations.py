"""Organization Management Endpoints for Multi-Tenant Control."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from backend.app.db.session import get_db
from backend.app.db.models.organization import Organization
from backend.app.db.models.user import User
from backend.app.schemas.organization import OrganizationResponse, OrganizationUsageResponse
from backend.app.api.deps import require_jwt_user, require_role

router = APIRouter()

@router.get("/current", response_model=OrganizationResponse)
def get_current_organization(user: User = Depends(require_jwt_user), db: Session = Depends(get_db)):
    """Returns organization profile, subscription tier, and operational parameters."""
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    return org

@router.get("/usage", response_model=OrganizationUsageResponse)
def get_organization_usage(user: User = Depends(require_jwt_user), db: Session = Depends(get_db)):
    """Returns monthly API quota utilization, remaining calls, and rate limit."""
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    if not org:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Organization not found.")
    
    remaining = max(0, org.monthly_request_quota - org.current_month_requests)
    pct = round((org.current_month_requests / org.monthly_request_quota) * 100.0, 2) if org.monthly_request_quota > 0 else 0.0

    return OrganizationUsageResponse(
        organization_id=org.id,
        organization_name=org.name,
        subscription_tier=org.subscription_tier,
        monthly_request_quota=org.monthly_request_quota,
        current_month_requests=org.current_month_requests,
        remaining_quota=remaining,
        quota_utilization_pct=pct,
        rate_limit_per_minute=org.rate_limit_per_minute,
        is_commercial_cleared=org.is_commercial_cleared,
        research_access_permitted=org.research_access_permitted,
        quota_reset_date=org.quota_reset_date
    )

@router.get("/members")
def get_organization_members(user: User = Depends(require_role(["OWNER", "ADMIN"])), db: Session = Depends(get_db)):
    """Lists all user accounts within the organization (Owner/Admin only)."""
    members = db.query(User).filter(User.organization_id == user.organization_id).all()
    return [
        {
            "id": m.id,
            "email": m.email,
            "full_name": m.full_name,
            "role": m.role,
            "is_active": m.is_active,
            "created_at": m.created_at
        }
        for m in members
    ]
