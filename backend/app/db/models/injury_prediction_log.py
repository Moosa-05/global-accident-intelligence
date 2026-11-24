from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON
from datetime import datetime
from backend.app.db.base import Base

class InjuryPredictionLog(Base):
    __tablename__ = "injury_prediction_logs"

    id = Column(String(36), primary_key=True, index=True)
    organization_id = Column(String(36), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    predicted_severity = Column(String(20), nullable=False, index=True)
    severity_level_code = Column(Integer, nullable=False)
    probabilities = Column(JSON, nullable=False)
    fatality_risk_alert = Column(Boolean, nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    latency_ms = Column(Float, nullable=False)
    features_payload = Column(JSON, nullable=True)
