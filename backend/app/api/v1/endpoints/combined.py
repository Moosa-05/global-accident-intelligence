from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.schemas.combined import CombinedPredictionRequest, CombinedPredictionResponse
from backend.app.schemas.auth import AuthContext
from backend.app.api.deps import get_db, get_combined_service, get_auth_context
from backend.app.services.combined_service import CombinedPredictionService
from backend.app.core.licensing import verify_combined_licensing_access
from backend.app.db.models.api_audit_log import ApiAuditLog

router = APIRouter()

@router.post("/combined", response_model=CombinedPredictionResponse, status_code=status.HTTP_200_OK)
def predict_combined_risk_and_severity(
    request: CombinedPredictionRequest,
    db: Session = Depends(get_db),
    service: CombinedPredictionService = Depends(get_combined_service),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    """
    Evaluates both real-time corridor sensor dynamics (accident occurrence risk)
    and atmospheric/road context (conditional traffic delay severity) into a unified risk assessment.
    Enforces dual-model licensing verification: non-commercial / research accounts only.
    """
    # Enforce licensing verification across both underlying constituent models
    verify_combined_licensing_access(auth_ctx)

    if not service.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Combined prediction service is not fully ready (one or more models unloaded)."
        )

    try:
        response = service.predict_combined(request, db_session=db)

        # Audit log for authenticated tenants (JWT or API Key)
        if auth_ctx and auth_ctx.auth_method not in ["demo", None]:
            audit = ApiAuditLog(
                organization_id=auth_ctx.organization_id,
                user_id=auth_ctx.user_id,
                api_key_id=auth_ctx.api_key_id,
                endpoint="/api/v1/predictions/combined",
                method="POST",
                model_id="combined_risk_and_severity",
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
            detail=f"Combined inference failed: {str(e)}"
        )
