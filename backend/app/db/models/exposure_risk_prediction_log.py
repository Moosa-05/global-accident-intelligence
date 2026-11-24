from sqlalchemy import Column, String, Float, Integer, DateTime
from datetime import datetime
from backend.app.db.base import Base

class ExposureRiskPredictionLog(Base):
    __tablename__ = "exposure_risk_prediction_logs"

    id = Column(String(36), primary_key=True, index=True)
    organization_id = Column(String(36), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    road_name = Column(String(100), nullable=True)
    road_class = Column(String(50), nullable=False, index=True)
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    annual_average_daily_flow = Column(Integer, nullable=False)
    link_length_km = Column(Float, nullable=False)
    annual_exposure_million_vkt = Column(Float, nullable=False)
    predicted_annual_crashes = Column(Float, nullable=False)
    predicted_crash_rate_per_million_vkt = Column(Float, nullable=False)
    risk_tier = Column(String(30), nullable=False, index=True)
    model_version = Column(String(50), nullable=False)
    latency_ms = Column(Float, nullable=False)
