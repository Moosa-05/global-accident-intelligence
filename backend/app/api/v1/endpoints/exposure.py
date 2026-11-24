"""Endpoint for Road Segment Annual Crash Frequency & Exposure-Adjusted Crash Rate."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.schemas.exposure_risk import (
    ExposureRiskPredictionRequest,
    ExposureRiskPredictionResponse
)
from backend.app.schemas.auth import AuthContext
from backend.app.services.exposure_risk_service import exposure_risk_service
from backend.app.api.deps import get_db, get_auth_context
from backend.app.core.licensing import verify_model_licensing_access
from backend.app.db.models.api_audit_log import ApiAuditLog

router = APIRouter()

@router.post(
    "/exposure-risk",
    response_model=ExposureRiskPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Estimate Road-Segment Crash Frequency & Exposure-Adjusted Crash Rate",
    description=(
        "Estimates annual expected injury-collision frequency E[Y] and normalized crash rate per Million Vehicle-Kilometres "
        "Travelled (MVKT) using fitted LightGBM / GLM Poisson models with log-exposure offsets (UK AADF & French TMJA exposure data). "
        "Commercial SaaS cleared under UK OGL v3.0 and France Licence Ouverte v2.0."
    )
)
def predict_exposure_risk(
    request: ExposureRiskPredictionRequest,
    db: Session = Depends(get_db),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    # Verify commercial licensing clearance (Permitted for all commercial & evaluation tiers)
    verify_model_licensing_access("exposure_risk_v1", auth_ctx)

    if not exposure_risk_service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exposure risk model service is not loaded or ready."
        )

    org_id = auth_ctx.organization_id if auth_ctx else None
    response = exposure_risk_service.predict(request, db_session=db, organization_id=org_id)

    # Audit log for authenticated tenants (JWT or API Key)
    if auth_ctx and auth_ctx.auth_method not in ["demo", None]:
        audit = ApiAuditLog(
            organization_id=auth_ctx.organization_id,
            user_id=auth_ctx.user_id,
            api_key_id=auth_ctx.api_key_id,
            endpoint="/api/v1/predictions/exposure-risk",
            method="POST",
            model_id="exposure_risk_v1",
            status_code=status.HTTP_200_OK,
            latency_ms=response.latency_ms
        )
        db.add(audit)
        db.commit()

    return response
