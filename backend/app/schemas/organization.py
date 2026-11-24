"""Organization & Usage Schemas."""
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel

class OrganizationResponse(BaseModel):
    id: str
    name: str
    slug: str
    subscription_tier: str
    subscription_status: str
    is_commercial_cleared: bool
    research_access_permitted: bool
    rate_limit_per_minute: int
    monthly_request_quota: int
    current_month_requests: int
    quota_reset_date: datetime
    created_at: datetime

class OrganizationUsageResponse(BaseModel):
    organization_id: str
    organization_name: str
    subscription_tier: str
    monthly_request_quota: int
    current_month_requests: int
    remaining_quota: int
    quota_utilization_pct: float
    rate_limit_per_minute: int
    is_commercial_cleared: bool
    research_access_permitted: bool
    quota_reset_date: datetime
