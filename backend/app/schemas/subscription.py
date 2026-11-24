"""Subscription and Entitlement Schemas."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class PlanDetails(BaseModel):
    tier: str
    name: str
    price_monthly_usd: float
    monthly_request_quota: int
    rate_limit_per_minute: int
    is_commercial_cleared: bool
    allowed_models: List[str]
    features: List[str]

class PlansListResponse(BaseModel):
    plans: List[PlanDetails]

class SubscriptionChangeRequest(BaseModel):
    new_tier: str  # FREE, DEVELOPER, PROFESSIONAL, BUSINESS, ENTERPRISE

class SubscriptionChangeResponse(BaseModel):
    organization_id: str
    previous_tier: str
    new_tier: str
    monthly_request_quota: int
    rate_limit_per_minute: int
    is_commercial_cleared: bool
    status: str
    message: str
