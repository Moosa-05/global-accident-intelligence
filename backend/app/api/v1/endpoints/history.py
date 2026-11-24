"""Unified Prediction History and Audit Log Endpoints with Strict Tenant Isolation."""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc, or_

from backend.app.db.session import get_db
from backend.app.db.models.prediction_log import PredictionLog
from backend.app.db.models.severity_prediction_log import SeverityPredictionLog
from backend.app.db.models.injury_prediction_log import InjuryPredictionLog
from backend.app.db.models.exposure_risk_prediction_log import ExposureRiskPredictionLog
from backend.app.db.models.api_audit_log import ApiAuditLog
from backend.app.schemas.auth import AuthContext
from backend.app.api.deps import get_auth_context

router = APIRouter()

@router.get("/predictions")
def get_prediction_history(
    limit: int = Query(50, ge=1, le=200),
    prediction_type: Optional[str] = Query(None, description="injury, exposure, risk, severity"),
    db: Session = Depends(get_db),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    """Returns chronologically ordered real prediction records isolated by organization."""
    records = []
    org_id = auth_ctx.organization_id if auth_ctx else None

    # Helper to apply tenant boundary filtering (strictly isolated to caller's verified session/org)
    def apply_tenant_filter(query, model):
        if org_id:
            return query.filter(model.organization_id == org_id)
        return query.filter(model.organization_id == "__unauthenticated_none__")

    # 1. Injury predictions
    if not prediction_type or prediction_type == "injury":
        q = apply_tenant_filter(db.query(InjuryPredictionLog), InjuryPredictionLog)
        injuries = q.order_by(desc(InjuryPredictionLog.created_at)).limit(limit).all()
        for i in injuries:
            records.append({
                "id": i.id,
                "type": "injury_severity",
                "model_id": "human_injury_severity_v1",
                "created_at": i.created_at.isoformat() if i.created_at else None,
                "latitude": i.latitude,
                "longitude": i.longitude,
                "summary": f"Severity: {i.predicted_severity} (Conditional on collision)",
                "result": {
                    "predicted_severity": i.predicted_severity,
                    "probabilities": i.probabilities,
                    "fatality_risk_alert": i.fatality_risk_alert
                },
                "latency_ms": i.latency_ms,
                "model_version": i.model_version
            })

    # 2. Exposure predictions
    if not prediction_type or prediction_type == "exposure":
        q = apply_tenant_filter(db.query(ExposureRiskPredictionLog), ExposureRiskPredictionLog)
        exposures = q.order_by(desc(ExposureRiskPredictionLog.created_at)).limit(limit).all()
        for e in exposures:
            records.append({
                "id": e.id,
                "type": "exposure_risk",
                "model_id": "exposure_risk_v1",
                "created_at": e.created_at.isoformat() if e.created_at else None,
                "latitude": e.latitude,
                "longitude": e.longitude,
                "summary": f"{e.road_name or e.road_class} - {e.predicted_annual_crashes:.1f} annual crashes ({e.risk_tier})",
                "result": {
                    "predicted_annual_crashes": e.predicted_annual_crashes,
                    "predicted_crash_rate_per_million_vkt": e.predicted_crash_rate_per_million_vkt,
                    "risk_tier": e.risk_tier,
                    "annual_exposure_million_vkt": e.annual_exposure_million_vkt
                },
                "latency_ms": e.latency_ms,
                "model_version": e.model_version
            })

    # 3. Corridor risk predictions
    if not prediction_type or prediction_type == "risk":
        q = apply_tenant_filter(db.query(PredictionLog), PredictionLog)
        risks = q.order_by(desc(PredictionLog.created_at)).limit(limit).all()
        for r in risks:
            records.append({
                "id": r.id,
                "type": "corridor_risk",
                "model_id": "accident_risk_v1",
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "latitude": None,
                "longitude": None,
                "summary": f"Corridor {r.road} Mile {r.mile} - 15-min Risk: {r.risk_score:.2%} ({r.risk_level})",
                "result": {
                    "risk_score": r.risk_score,
                    "risk_level": r.risk_level,
                    "accident_alert": r.accident_alert
                },
                "latency_ms": r.latency_ms,
                "model_version": r.model_version
            })

    # 4. Severity predictions
    if not prediction_type or prediction_type == "severity":
        q = apply_tenant_filter(db.query(SeverityPredictionLog), SeverityPredictionLog)
        sevs = q.order_by(desc(SeverityPredictionLog.created_at)).limit(limit).all()
        for s in sevs:
            records.append({
                "id": s.id,
                "type": "traffic_severity",
                "model_id": "accident_severity_v2",
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "latitude": s.latitude,
                "longitude": s.longitude,
                "summary": f"Traffic Delay Level {s.predicted_severity} ({s.severity_level})",
                "result": {
                    "predicted_severity": s.predicted_severity,
                    "severity_level": s.severity_level,
                    "probabilities": s.probabilities
                },
                "latency_ms": s.latency_ms,
                "model_version": s.model_version
            })

    # Sort all by created_at desc and slice
    records.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return {"total": len(records), "records": records[:limit]}

@router.get("/audit-logs")
def get_audit_logs(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    auth_ctx: AuthContext = Depends(get_auth_context)
):
    """Returns API audit trail logs strictly isolated to the caller's organization."""
    query = db.query(ApiAuditLog)
    if auth_ctx and auth_ctx.organization_id:
        query = query.filter(ApiAuditLog.organization_id == auth_ctx.organization_id)
    else:
        query = query.filter(ApiAuditLog.organization_id == "__unauthenticated_none__")
    logs = query.order_by(desc(ApiAuditLog.created_at)).limit(limit).all()
    return [
        {
            "id": l.id,
            "endpoint": l.endpoint,
            "method": l.method,
            "model_id": l.model_id,
            "status_code": l.status_code,
            "latency_ms": l.latency_ms,
            "ip_address": l.ip_address,
            "created_at": l.created_at.isoformat() if l.created_at else None
        }
        for l in logs
    ]
