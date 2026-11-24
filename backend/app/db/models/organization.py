"""Organization Model for Multi-Tenant SaaS Isolation."""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, Integer, DateTime
from sqlalchemy.orm import relationship

from backend.app.db.base import Base

class Organization(Base):
    __tablename__ = "organizations"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    slug = Column(String(100), unique=True, index=True, nullable=False)
    
    # Subscription & Entitlements
    subscription_tier = Column(String(50), default="FREE", nullable=False)  # FREE, DEVELOPER, PROFESSIONAL, BUSINESS, ENTERPRISE
    subscription_status = Column(String(50), default="ACTIVE", nullable=False)  # ACTIVE, PAST_DUE, CANCELED, TRIALING
    is_commercial_cleared = Column(Boolean, default=False, nullable=False)  # True for paid tiers (Dev/Pro/Bus/Ent)
    research_access_permitted = Column(Boolean, default=False, nullable=False)  # True for research/academic accounts
    
    # Rate Limits & Quotas
    rate_limit_per_minute = Column(Integer, default=10, nullable=False)
    monthly_request_quota = Column(Integer, default=100, nullable=False)
    current_month_requests = Column(Integer, default=0, nullable=False)
    quota_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Payment / Billing metadata
    billing_email = Column(String(255), nullable=True)
    payment_customer_id = Column(String(100), nullable=True)
    payment_subscription_id = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="organization", cascade="all, delete-orphan")
    audit_logs = relationship("ApiAuditLog", back_populates="organization", cascade="all, delete-orphan")
