"""Authentication Endpoints for User Registration, Login, Token Refresh, and Profile."""
import re
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.config import settings
from backend.app.core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token
)
from backend.app.db.models.organization import Organization
from backend.app.db.models.user import User
from backend.app.schemas.auth import (
    UserRegisterRequest,
    UserLoginRequest,
    TokenRefreshRequest,
    TokenResponse,
    UserResponse
)
from backend.app.services.subscription_service import PLAN_CATALOG
from backend.app.api.deps import require_jwt_user

router = APIRouter()

def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[-\s]+", "-", slug)

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_organization_and_user(req: UserRegisterRequest, db: Session = Depends(get_db)):
    """Registers a new tenant organization and initial owner account."""
    # Check if user email already exists
    existing_user = db.query(User).filter(User.email == req.email.lower()).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user account with this email address already exists."
        )

    # Build unique organization slug
    base_slug = slugify(req.organization_name)
    slug = base_slug
    idx = 1
    while db.query(Organization).filter(Organization.slug == slug).first():
        slug = f"{base_slug}-{idx}"
        idx += 1

    tier = req.subscription_tier.upper() if req.subscription_tier else "FREE"
    if tier not in PLAN_CATALOG:
        tier = "FREE"
    plan_info = PLAN_CATALOG[tier]

    # Create Organization
    org = Organization(
        name=req.organization_name,
        slug=slug,
        subscription_tier=tier,
        subscription_status="ACTIVE",
        is_commercial_cleared=plan_info["is_commercial_cleared"],
        research_access_permitted=bool(req.research_access_requested),
        rate_limit_per_minute=plan_info["rate_limit_per_minute"],
        monthly_request_quota=plan_info["monthly_request_quota"],
        current_month_requests=0,
        billing_email=req.email.lower()
    )
    db.add(org)
    db.flush()

    # Create Owner User
    user = User(
        organization_id=org.id,
        email=req.email.lower(),
        hashed_password=get_password_hash(req.password),
        full_name=req.full_name or req.organization_name,
        role="OWNER",
        is_active=True,
        is_verified=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    db.refresh(org)

    # Generate tokens
    token_data = {"sub": user.id, "org_id": org.id, "role": user.role}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return {
        "message": "Organization and user account successfully created.",
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "organization_id": org.id
        },
        "organization": {
            "id": org.id,
            "name": org.name,
            "slug": org.slug,
            "subscription_tier": org.subscription_tier,
            "is_commercial_cleared": org.is_commercial_cleared,
            "research_access_permitted": org.research_access_permitted
        },
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in_seconds": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    }

@router.post("/login", response_model=TokenResponse)
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticates email and password, returning JWT access and refresh tokens."""
    user = db.query(User).filter(User.email == req.email.lower()).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password."
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated.")

    token_data = {"sub": user.id, "org_id": user.organization_id, "role": user.role}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@router.post("/refresh", response_model=TokenResponse)
def refresh_token(req: TokenRefreshRequest, db: Session = Depends(get_db)):
    """Issues a new access token using a valid refresh token."""
    payload = decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive.")

    token_data = {"sub": user.id, "org_id": user.organization_id, "role": user.role}
    new_access = create_access_token(token_data)
    new_refresh = create_refresh_token(token_data)

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_refresh,
        token_type="bearer",
        expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

@router.get("/me")
def get_current_user_profile(user: User = Depends(require_jwt_user), db: Session = Depends(get_db)):
    """Returns profile and organization status for current authenticated user."""
    org = db.query(Organization).filter(Organization.id == user.organization_id).first()
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "created_at": user.created_at
        },
        "organization": {
            "id": org.id,
            "name": org.name,
            "slug": org.slug,
            "subscription_tier": org.subscription_tier,
            "subscription_status": org.subscription_status,
            "is_commercial_cleared": org.is_commercial_cleared,
            "research_access_permitted": org.research_access_permitted,
            "rate_limit_per_minute": org.rate_limit_per_minute,
            "monthly_request_quota": org.monthly_request_quota,
            "current_month_requests": org.current_month_requests
        }
    }
