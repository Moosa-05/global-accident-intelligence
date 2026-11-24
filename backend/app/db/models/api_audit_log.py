"""API Request Audit Log Model for Telemetry Tracking & Compliance."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text

from sqlalchemy.orm import relationship
from backend.app.db.base import Base

class ApiAuditLog(Base):
    __tablename__ = "api_audit_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(36), nullable=True)
    api_key_id = Column(String(36), nullable=True)
    
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    model_id = Column(String(100), nullable=True)
    status_code = Column(Integer, nullable=False)
    latency_ms = Column(Float, nullable=False)
    ip_address = Column(String(45), nullable=True)
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    organization = relationship("Organization", back_populates="audit_logs")
