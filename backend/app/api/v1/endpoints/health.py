from fastapi import APIRouter, Depends, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any
from backend.app.schemas.health import HealthResponse
from backend.app.core.config import settings
from backend.app.api.deps import (
    get_db,
    get_prediction_service,
    get_severity_service,
    get_injury_severity_service,
    get_exposure_risk_service
)
from backend.app.services.prediction_service import RiskPredictionService
from backend.app.services.severity_service import SeverityPredictionService
from backend.app.services.injury_severity_service import InjurySeverityService
from backend.app.services.exposure_risk_service import ExposureRiskService

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check(
    db: Session = Depends(get_db),
    risk_svc: RiskPredictionService = Depends(get_prediction_service),
    sev_svc: SeverityPredictionService = Depends(get_severity_service),
    inj_svc: InjurySeverityService = Depends(get_injury_severity_service),
    exp_svc: ExposureRiskService = Depends(get_exposure_risk_service)
):
    try:
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    models_ready = {
        "accident_risk": risk_svc.is_ready,
        "accident_severity": sev_svc.is_ready,
        "human_injury_severity": inj_svc.is_ready,
        "exposure_risk": exp_svc.is_ready
    }
    all_ready = all(models_ready.values()) and db_status == "connected"

    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        environment=settings.ENVIRONMENT,
        version=settings.VERSION,
        database=db_status,
        models_ready=models_ready,
        model_loaded=all_ready,
        model_version=f"risk:{risk_svc.model_version}|sev:{sev_svc.model_version}|inj:{inj_svc.model_version}|exp:{exp_svc.model_version}"
    )

@router.get("/health/live")
def liveness_probe() -> Dict[str, Any]:
    """Lightweight Kubernetes/Docker liveness probe."""
    return {
        "status": "alive",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }

@router.get("/health/ready")
def readiness_probe(
    response: Response,
    db: Session = Depends(get_db),
    risk_svc: RiskPredictionService = Depends(get_prediction_service),
    sev_svc: SeverityPredictionService = Depends(get_severity_service),
    inj_svc: InjurySeverityService = Depends(get_injury_severity_service),
    exp_svc: ExposureRiskService = Depends(get_exposure_risk_service)
) -> Dict[str, Any]:
    """Production readiness probe validating DB connectivity and model readiness."""
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
        db_status = "connected"
    except Exception as e:
        db_status = f"unhealthy: {e}"

    models_ready = {
        "accident_risk": risk_svc.is_ready,
        "accident_severity": sev_svc.is_ready,
        "human_injury_severity": inj_svc.is_ready,
        "exposure_risk": exp_svc.is_ready
    }
    is_ready = db_connected and all(models_ready.values())

    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "ready": False,
            "database": db_status,
            "models_ready": models_ready
        }

    return {
        "status": "ready",
        "ready": True,
        "database": db_status,
        "models_ready": models_ready,
        "version": settings.VERSION
    }
