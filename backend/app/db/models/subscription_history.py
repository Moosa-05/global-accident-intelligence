"""Subscription Change History Model."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, ForeignKey, Text

from backend.app.db.base import Base

class SubscriptionHistory(Base):
    __tablename__ = "subscription_history"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    previous_tier = Column(String(50), nullable=False)
    new_tier = Column(String(50), nullable=False)
    reason = Column(String(100), default="UPGRADE", nullable=False)
    notes = Column(Text, nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
