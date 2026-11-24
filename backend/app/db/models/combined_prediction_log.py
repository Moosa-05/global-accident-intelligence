import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.db.base import Base

class CombinedPredictionLog(Base):
    __tablename__ = "combined_prediction_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    accident_alert: Mapped[bool] = mapped_column(Boolean, nullable=False)
    predicted_severity: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    expected_severity_index: Mapped[float] = mapped_column(Float, nullable=False)
    joint_risk_impact_score: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    payload_summary: Mapped[dict] = mapped_column(JSON, nullable=True)
