"""SaaS Subscription & Entitlement Service.
Manages tiers: FREE, DEVELOPER, PROFESSIONAL, BUSINESS, ENTERPRISE.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from backend.app.schemas.subscription import PlanDetails, PlansListResponse
from backend.app.db.models.organization import Organization
from backend.app.db.models.subscription_history import SubscriptionHistory

PLAN_CATALOG: Dict[str, Dict[str, Any]] = {
    "FREE": {
        "tier": "FREE",
        "name": "Free Community & Evaluation",
        "price_monthly_usd": 0.0,
        "monthly_request_quota": 100,
        "rate_limit_per_minute": 10,
        "is_commercial_cleared": False,
        "allowed_models": ["human_injury_severity_v1", "exposure_risk_v1"],
        "features": [
            "100 requests per month",
            "10 requests per minute rate limit",
            "Access to OGL / Licence Ouverte commercial models for evaluation",
            "Non-commercial evaluation only"
        ]
    },
    "DEVELOPER": {
        "tier": "DEVELOPER",
        "name": "Developer Tier",
        "price_monthly_usd": 99.0,
        "monthly_request_quota": 5000,
        "rate_limit_per_minute": 60,
        "is_commercial_cleared": True,
        "allowed_models": ["human_injury_severity_v1", "exposure_risk_v1"],
        "features": [
            "5,000 requests per month",
            "60 requests per minute rate limit",
            "Full commercial SaaS deployment clearance",
            "Global WGS84 & ERA5 hourly weather features",
            "Email developer support"
        ]
    },
    "PROFESSIONAL": {
        "tier": "PROFESSIONAL",
        "name": "Professional Tier",
        "price_monthly_usd": 499.0,
        "monthly_request_quota": 50000,
        "rate_limit_per_minute": 300,
        "is_commercial_cleared": True,
        "allowed_models": ["human_injury_severity_v1", "exposure_risk_v1"],
        "features": [
            "50,000 requests per month",
            "300 requests per minute rate limit",
            "Commercial SaaS deployment clearance",
            "Continuous multi-country exposure risk modeling",
            "Priority API SLA (99.9%)",
            "Dedicated Slack channel support"
        ]
    },
    "BUSINESS": {
        "tier": "BUSINESS",
        "name": "Business Tier",
        "price_monthly_usd": 2499.0,
        "monthly_request_quota": 500000,
        "rate_limit_per_minute": 1200,
        "is_commercial_cleared": True,
        "allowed_models": ["human_injury_severity_v1", "exposure_risk_v1"],
        "features": [
            "500,000 requests per month",
            "1,200 requests per minute rate limit",
            "Full commercial licensing clearance",
            "Dedicated inference infrastructure",
            "Custom webhooks and telematic batch pipelines",
            "24/7 dedicated support"
        ]
    },
    "ENTERPRISE": {
        "tier": "ENTERPRISE",
        "name": "Enterprise Global",
        "price_monthly_usd": 7500.0,
        "monthly_request_quota": 5000000,
        "rate_limit_per_minute": 5000,
        "is_commercial_cleared": True,
        "allowed_models": ["human_injury_severity_v1", "exposure_risk_v1"],
        "features": [
            "5,000,000+ requests per month",
            "5,000 requests per minute rate limit",
            "Unlimited API keys and users",
            "Optional Academic/Research dual licensing rider",
            "On-premise / private VPC deployment options",
            "Dedicated Account Director and custom SLA"
        ]
    }
}

class SubscriptionService:
    def __init__(self):
        pass

    def get_plans(self) -> PlansListResponse:
        plans = [PlanDetails(**p) for p in PLAN_CATALOG.values()]
        return PlansListResponse(plans=plans)

    def change_subscription(self, org: Organization, new_tier: str, db: Session) -> Dict[str, Any]:
        tier_key = new_tier.upper()
        if tier_key not in PLAN_CATALOG:
            raise ValueError(f"Invalid subscription tier: {new_tier}. Available: {list(PLAN_CATALOG.keys())}")
        
        plan_info = PLAN_CATALOG[tier_key]
        prev_tier = org.subscription_tier

        # Apply plan entitlements to organization directly
        org.subscription_tier = tier_key
        org.subscription_status = "ACTIVE"
        org.is_commercial_cleared = plan_info["is_commercial_cleared"]
        org.monthly_request_quota = plan_info["monthly_request_quota"]
        org.rate_limit_per_minute = plan_info["rate_limit_per_minute"]

        # Log change
        history_entry = SubscriptionHistory(
            organization_id=org.id,
            previous_tier=prev_tier,
            new_tier=tier_key,
            reason="PLAN_CHANGE",
            notes=f"Changed subscription from {prev_tier} to {tier_key}"
        )
        db.add(history_entry)
        db.commit()
        db.refresh(org)

        return {
            "organization_id": org.id,
            "previous_tier": prev_tier,
            "new_tier": tier_key,
            "monthly_request_quota": org.monthly_request_quota,
            "rate_limit_per_minute": org.rate_limit_per_minute,
            "is_commercial_cleared": org.is_commercial_cleared,
            "status": "ACTIVE",
            "message": f"Successfully switched organization to {plan_info['name']}."
        }
