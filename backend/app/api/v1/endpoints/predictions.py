from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.schemas.prediction import RiskPredictionRequest, RiskPredictionResponse
from backend.app.schemas.auth import AuthContext
from backend.app.api.deps import get_db, get_prediction_service, get_auth_context
from backend.app.services.prediction_service import RiskPredictionService
from backend.app.core.licensing import verify_model_licensing_access
from backend.app.db.models.api_audit_log import ApiAuditLog

router = APIRouter()

@router.post("/risk", response_model=RiskPredictionResponse, status_code=status.HTTP_200_OK)
def predict_accident_risk(
    request: RiskPredictionRequest,
    db: Session = Depends(get_db),
    service: RiskPredictionService = Depends(get_prediction_service),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    """
    Evaluates real-time 15-minute traffic sensor dynamics to forecast
    accident likelihood and determine operational alert triggers.
    Enforces licensing restriction: Academic/Research benchmark only.
    """
    # Enforce endpoint-level commercial licensing
    verify_model_licensing_access("accident_risk_v1", auth_ctx)

    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Risk prediction model is not ready."
        )

    try:
        org_id = auth_ctx.organization_id if auth_ctx else None
        response = service.predict_risk(request, db_session=db, organization_id=org_id)
        
        # Audit log for authenticated tenants (JWT or API Key)
        if auth_ctx and auth_ctx.auth_method not in ["demo", None]:
            audit = ApiAuditLog(
                organization_id=auth_ctx.organization_id,
                user_id=auth_ctx.user_id,
                api_key_id=auth_ctx.api_key_id,
                endpoint="/api/v1/predictions/risk",
                method="POST",
                model_id="accident_risk_v1",
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
            detail=f"Inference processing failed: {str(e)}"
        )
