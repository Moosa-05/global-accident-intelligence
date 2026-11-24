from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.schemas.severity import SeverityPredictionRequest, SeverityPredictionResponse
from backend.app.schemas.auth import AuthContext
from backend.app.api.deps import get_db, get_severity_service, get_auth_context
from backend.app.services.severity_service import SeverityPredictionService
from backend.app.core.licensing import verify_model_licensing_access
from backend.app.db.models.api_audit_log import ApiAuditLog

router = APIRouter()

@router.post("/severity", response_model=SeverityPredictionResponse, status_code=status.HTTP_200_OK)
def predict_accident_severity(
    request: SeverityPredictionRequest,
    db: Session = Depends(get_db),
    service: SeverityPredictionService = Depends(get_severity_service),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    """
    Evaluates global geographic, atmospheric, and road-infrastructure context
    to estimate traffic-impact delay severity (classes 1 through 4) conditional on an incident.
    Enforces licensing restriction: Non-commercial research only (CC BY-NC-SA 4.0).
    """
    # Enforce endpoint-level commercial licensing
    verify_model_licensing_access("accident_severity_v2", auth_ctx)

    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Severity prediction model is not ready."
        )

    try:
        org_id = auth_ctx.organization_id if auth_ctx else None
        response = service.predict_severity(request, db_session=db, organization_id=org_id)

        # Audit log for authenticated tenants (JWT or API Key)
        if auth_ctx and auth_ctx.auth_method not in ["demo", None]:
            audit = ApiAuditLog(
                organization_id=auth_ctx.organization_id,
                user_id=auth_ctx.user_id,
                api_key_id=auth_ctx.api_key_id,
                endpoint="/api/v1/predictions/severity",
                method="POST",
                model_id="accident_severity_v2",
                status_code=status.HTTP_200_OK,
                latency_ms=response.latency_ms
            )
            db.add(audit)
            db.commit()

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Severity inference processing failed: {str(e)}"
        )
