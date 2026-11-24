import hmac
import hashlib
import uuid
from typing import Generator, Optional, List
from datetime import datetime
from fastapi import Request, Response, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.core.config import settings
from backend.app.core.security import decode_token, verify_api_key
from backend.app.core.rate_limiter import rate_limiter
from backend.app.db.models.organization import Organization
from backend.app.db.models.user import User
from backend.app.db.models.api_key import ApiKey
from backend.app.schemas.auth import AuthContext
from backend.app.services.prediction_service import prediction_service, RiskPredictionService
from backend.app.services.severity_service import severity_service, SeverityPredictionService
from backend.app.services.combined_service import combined_service, CombinedPredictionService
from backend.app.services.injury_severity_service import injury_severity_service, InjurySeverityService
from backend.app.services.exposure_risk_service import exposure_risk_service, ExposureRiskService
from backend.app.services.subscription_service import SubscriptionService

security_bearer = HTTPBearer(auto_error=False)

def get_prediction_service() -> RiskPredictionService:
    return prediction_service

def get_severity_service() -> SeverityPredictionService:
    return severity_service

def get_combined_service() -> CombinedPredictionService:
    return combined_service

def get_injury_severity_service() -> InjurySeverityService:
    return injury_severity_service

def get_exposure_risk_service() -> ExposureRiskService:
    return exposure_risk_service

def get_subscription_service() -> SubscriptionService:
    return SubscriptionService()

def generate_signed_session_token(session_uuid: str, secret_key: str) -> str:
    """Generates an HMAC-SHA256 signature for a session UUID."""
    sig = hmac.new(secret_key.encode("utf-8"), session_uuid.encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    return f"{session_uuid}.{sig}"

def verify_session_token(token: Optional[str], secret_key: str) -> Optional[str]:
    """Validates an HMAC-signed session token. Returns session_uuid if valid, None if tampered or invalid."""
    if not token or "." not in token:
        return None
    parts = token.split(".")
    if len(parts) != 2:
        return None
    session_uuid, received_sig = parts[0], parts[1]
    if not session_uuid or not received_sig:
        return None
    expected_sig = hmac.new(secret_key.encode("utf-8"), session_uuid.encode("utf-8"), hashlib.sha256).hexdigest()[:24]
    if hmac.compare_digest(received_sig, expected_sig):
        return session_uuid
    return None

def get_client_ip(request: Request) -> str:
    """Resolves client IP with reverse-proxy header support."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"

def get_auth_context(
    request: Request,
    response: Response,
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID"),
    db: Session = Depends(get_db)
) -> AuthContext:
    """Resolves authentication context from JWT bearer token, X-API-Key header, or secure anonymous session.
    Provides cryptographically verified, isolated session contexts for public demonstration visitors.
    Enforces IP-based and organization-level rate limits.
    """
    auth_ctx = None

    # 1. Check API Key
    if x_api_key:
        # Extract prefix: e.g. agy_live_a1b2c3d4e5f6_secret...
        parts = x_api_key.split("_")
        if len(parts) >= 3:
            key_prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"
            api_key = db.query(ApiKey).filter(ApiKey.key_prefix == key_prefix, ApiKey.is_active == True).first()
            if api_key and verify_api_key(x_api_key, api_key.key_prefix, api_key.hashed_secret):
                if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key has expired.")
                
                org = db.query(Organization).filter(Organization.id == api_key.organization_id).first()
                if org:
                    # Rate limit check
                    if settings.ENABLE_RATE_LIMITING:
                        allowed, count, retry_after = rate_limiter.is_allowed(f"key_{api_key.id}", org.rate_limit_per_minute)
                        if not allowed:
                            raise HTTPException(
                                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                detail=f"Rate limit exceeded ({org.rate_limit_per_minute} req/min). Retry after {retry_after}s.",
                                headers={"Retry-After": str(retry_after)}
                            )
                    
                    # Quota check
                    if org.current_month_requests >= org.monthly_request_quota:
                        raise HTTPException(
                            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail=f"Monthly quota exceeded ({org.monthly_request_quota} req/month for {org.subscription_tier} tier). Please upgrade your subscription."
                        )

                    # Update key usage
                    api_key.last_used_at = datetime.utcnow()
                    api_key.total_requests += 1
                    org.current_month_requests += 1
                    db.commit()

                    auth_ctx = AuthContext(
                        organization_id=org.id,
                        organization_name=org.name,
                        subscription_tier=org.subscription_tier,
                        is_commercial_cleared=org.is_commercial_cleared,
                        research_access_permitted=org.research_access_permitted,
                        scopes=api_key.scopes or [],
                        auth_method="api_key",
                        api_key_id=api_key.id
                    )
            else:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    # 2. Check JWT Bearer Token
    elif bearer and bearer.credentials:
        payload = decode_token(bearer.credentials)
        if not payload or payload.get("type") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token.")
        
        user_id = payload.get("sub")
        org_id = payload.get("org_id")
        user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
        org = db.query(Organization).filter(Organization.id == org_id).first()
        if not user or not org:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User or organization not found.")

        # Rate limit check on org
        if settings.ENABLE_RATE_LIMITING:
            allowed, count, retry_after = rate_limiter.is_allowed(f"org_{org.id}", org.rate_limit_per_minute)
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded ({org.rate_limit_per_minute} req/min). Retry after {retry_after}s.",
                    headers={"Retry-After": str(retry_after)}
                )

        auth_ctx = AuthContext(
            user_id=user.id,
            email=user.email,
            role=user.role,
            organization_id=org.id,
            organization_name=org.name,
            subscription_tier=org.subscription_tier,
            is_commercial_cleared=org.is_commercial_cleared,
            research_access_permitted=org.research_access_permitted,
            scopes=["*"],
            auth_method="jwt"
        )

    # 3. Default Public Demonstration Context (cryptographically verified session isolation)
    if auth_ctx is None:
        client_ip = get_client_ip(request)

        # 3a. IP-based rate limiting on public demonstration endpoints (60 req/min per IP)
        if settings.ENABLE_RATE_LIMITING:
            allowed, count, retry_after = rate_limiter.is_allowed(f"demo_ip_{client_ip}", 60)
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded for demonstration IP ({60} req/min). Retry after {retry_after}s.",
                    headers={"Retry-After": str(retry_after)}
                )

        # 3b. Cryptographically verified anonymous session token
        session_uuid = verify_session_token(x_session_id, settings.SECRET_KEY)
        if not session_uuid:
            # Untrusted or missing token: generate a fresh verified session
            session_uuid = uuid.uuid4().hex[:16]
            signed_token = generate_signed_session_token(session_uuid, settings.SECRET_KEY)
        else:
            signed_token = x_session_id

        # Return verified session token to the browser client
        if response is not None:
            response.headers["X-Session-ID"] = signed_token

        auth_ctx = AuthContext(
            organization_id=f"demo_{session_uuid}",
            organization_name="Demonstration Session",
            subscription_tier="DEMO",
            is_commercial_cleared=True,
            research_access_permitted=True,  # Allows evaluation across all models in demo
            scopes=["*"],
            auth_method="demo"
        )

    return auth_ctx

def require_jwt_user(
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    db: Session = Depends(get_db)
) -> User:
    """Strictly requires a valid JWT access token and active User."""
    if not bearer or not bearer.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication credentials required.")
    
    payload = decode_token(bearer.credentials)
    if not payload or payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired access token.")
    
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account not found or deactivated.")
    return user

def require_role(allowed_roles: List[str]):
    def role_checker(user: User = Depends(require_jwt_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required role: {allowed_roles}, your role: {user.role}"
            )
        return user
    return role_checker
