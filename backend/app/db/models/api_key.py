"""API Key Model for Organization Telematics & Client Applications."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, Integer, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from backend.app.db.base import Base

class ApiKey(Base):
    __tablename__ = "api_keys"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    
    name = Column(String(100), nullable=False)
    key_prefix = Column(String(32), index=True, nullable=False)  # e.g., agy_live_a1b2c3d4e5f6
    hashed_secret = Column(String(128), nullable=False)
    scopes = Column(JSON, default=list, nullable=False)  # ["predict:risk", "predict:injury", etc.]
    
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    total_requests = Column(Integer, default=0, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    # Relationships
    organization = relationship("Organization", back_populates="api_keys")
