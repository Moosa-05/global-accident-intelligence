"""Endpoint for Human Injury Severity Prediction (3-Class: SLIGHT, SERIOUS, FATAL)."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.schemas.injury_severity import (
    InjurySeverityPredictionRequest,
    InjurySeverityPredictionResponse
)
from backend.app.schemas.auth import AuthContext
from backend.app.services.injury_severity_service import injury_severity_service
from backend.app.api.deps import get_db, get_auth_context
from backend.app.core.licensing import verify_model_licensing_access
from backend.app.db.models.api_audit_log import ApiAuditLog

router = APIRouter()

@router.post(
    "/injury-severity",
    response_model=InjurySeverityPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Human Physiological Injury Severity (3-Class)",
    description=(
        "Predicts collision-level physiological trauma severity (SLIGHT, SERIOUS, FATAL) using the "
        "multi-country European road safety models (UK STATS19 & France BAAC). "
        "Commercial SaaS cleared under UK OGL v3.0 and France Licence Ouverte v2.0."
    )
)
def predict_injury_severity(
    request: InjurySeverityPredictionRequest,
    db: Session = Depends(get_db),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    # Verify commercial licensing clearance (Permitted for all commercial & evaluation tiers)
    verify_model_licensing_access("human_injury_severity_v1", auth_ctx)

    if not injury_severity_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human injury severity model service is not loaded or ready."
        )

    org_id = auth_ctx.organization_id if auth_ctx else None
    response = injury_severity_service.predict(request, db_session=db, organization_id=org_id)

    # Audit log for authenticated tenants (JWT or API Key)
    if auth_ctx and auth_ctx.auth_method not in ["demo", None]:
        audit = ApiAuditLog(
            organization_id=auth_ctx.organization_id,
            user_id=auth_ctx.user_id,
            api_key_id=auth_ctx.api_key_id,
            endpoint="/api/v1/predictions/injury-severity",
            method="POST",
            model_id="human_injury_severity_v1",
            status_code=status.HTTP_200_OK,
            latency_ms=response.latency_ms
        )
        db.add(audit)
        db.commit()

    return response
