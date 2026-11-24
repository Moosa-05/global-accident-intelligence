import time
from datetime import datetime
from typing import Dict, Any

from backend.app.core.logging import logger
from backend.app.schemas.combined import (
    CombinedPredictionRequest,
    CombinedPredictionResponse,
    RiskSummary,
    SeveritySummary
)
from backend.app.services.prediction_service import prediction_service
from backend.app.services.severity_service import severity_service
from backend.app.db.models.combined_prediction_log import CombinedPredictionLog

class CombinedPredictionService:
    def __init__(self):
        self.risk_service = prediction_service
        self.severity_service = severity_service

    @property
    def is_ready(self) -> bool:
        return self.risk_service.is_ready and self.severity_service.is_ready

    def predict_combined(self, request: CombinedPredictionRequest, db_session = None) -> CombinedPredictionResponse:
        t0 = time.time()
        
        # 1. Evaluate Incident Occurrence Risk
        risk_res = self.risk_service.predict_risk(request.risk_inputs, db_session=None)
        
        # 2. Evaluate Conditional Severity Impact
        sev_res = self.severity_service.predict_severity(request.severity_inputs, db_session=None)

        # 3. Compute Composite Metrics
        probs = sev_res.severity_probabilities
        # Expected value of severity: 1*P(1) + 2*P(2) + 3*P(3) + 4*P(4)
        expected_sev = (
            1.0 * probs["class_1"] +
            2.0 * probs["class_2"] +
            3.0 * probs["class_3"] +
            4.0 * probs["class_4"]
        )

        # Joint impact: probability of accident * normalized expected severity impact [0, 1]
        joint_score = float(risk_res.risk_score * (expected_sev / 4.0))
        latency_ms = round((time.time() - t0) * 1000, 2)

        response = CombinedPredictionResponse(
            accident_risk=RiskSummary(
                risk_score=risk_res.risk_score,
                risk_level=risk_res.risk_level,
                accident_alert=risk_res.accident_alert,
                threshold_used=risk_res.threshold_used
            ),
            conditional_severity=SeveritySummary(
                predicted_severity=sev_res.predicted_severity,
                severity_level=sev_res.severity_level,
                severity_probabilities=sev_res.severity_probabilities
            ),
            expected_severity_index=round(expected_sev, 3),
            joint_risk_impact_score=round(joint_score, 4),
            model_versions={
                "risk_model": risk_res.model_version,
                "severity_model": sev_res.model_version
            },
            latency_ms=latency_ms,
            timestamp=datetime.utcnow().isoformat(),
            geographic_metadata=sev_res.geographic_metadata
        )

        # Log combined record to DB
        if db_session is not None:
            try:
                log_entry = CombinedPredictionLog(
                    id=response.prediction_id,
                    risk_score=risk_res.risk_score,
                    accident_alert=risk_res.accident_alert,
                    predicted_severity=sev_res.predicted_severity,
                    expected_severity_index=response.expected_severity_index,
                    joint_risk_impact_score=response.joint_risk_impact_score,
                    latency_ms=latency_ms,
                    payload_summary={
                        "road": request.risk_inputs.road,
                        "mile": request.risk_inputs.mile,
                        "latitude": request.severity_inputs.latitude,
                        "longitude": request.severity_inputs.longitude
                    }
                )
                db_session.add(log_entry)
                db_session.commit()
            except Exception as e:
                logger.error(f"Failed to log combined prediction to DB: {e}")
                db_session.rollback()

        return response

combined_service = CombinedPredictionService()
