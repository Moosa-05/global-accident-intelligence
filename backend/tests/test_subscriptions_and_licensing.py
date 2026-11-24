"""Tests for Subscription Plans, Plan Upgrades, and Endpoint-Level Model Licensing Enforcement."""
import pytest
from fastapi.testclient import TestClient

def test_list_subscription_plans(client: TestClient):
    r = client.get("/api/v1/subscriptions/plans")
    assert r.status_code == 200
    data = r.json()
    assert "plans" in data
    tiers = [p["tier"] for p in data["plans"]]
    assert "FREE" in tiers
    assert "DEVELOPER" in tiers
    assert "PROFESSIONAL" in tiers
    assert "BUSINESS" in tiers
    assert "ENTERPRISE" in tiers

def test_change_subscription_plan(client: TestClient):
    # Register an organization
    reg_res = client.post("/api/v1/auth/register", json={
        "email": "cto@transitfleet.com",
        "password": "SecurePassword123!",
        "organization_name": "Transit Fleet Solutions",
        "subscription_tier": "FREE"
    })
    token = reg_res.json()["tokens"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Change to PROFESSIONAL tier
    change_res = client.post("/api/v1/subscriptions/change-plan", headers=headers, json={
        "new_tier": "PROFESSIONAL"
    })
    assert change_res.status_code == 200
    sub_data = change_res.json()
    assert sub_data["new_tier"] == "PROFESSIONAL"
    assert sub_data["monthly_request_quota"] == 50000
    assert sub_data["rate_limit_per_minute"] == 300
    assert sub_data["is_commercial_cleared"] is True

def test_commercial_licensing_enforcement_at_prediction_endpoints(client: TestClient):
    """Critical SaaS requirement:
    Commercial accounts must be blocked from accessing non-commercial research models:
    - accident_severity_v2 (CC BY-NC-SA 4.0 Non-Commercial)
    - accident_risk_v1 (Research Benchmark)
    - combined endpoint (incorporating non-commercial models)
    And granted access to commercially cleared models:
    - human_injury_severity_v1 (OGL v3.0 & Licence Ouverte v2.0)
    - exposure_risk_v1 (OGL v3.0 & Licence Ouverte v2.0)
    """
    # 1. Register a Commercial Tenant (DEVELOPER tier, commercial=True, research=False)
    reg_res = client.post("/api/v1/auth/register", json={
        "email": "corp@commercial-insurance.com",
        "password": "SecurePassword123!",
        "organization_name": "Commercial InsurTech Corp",
        "subscription_tier": "DEVELOPER",
        "research_access_requested": False
    })
    token = reg_res.json()["tokens"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # A. Commercial tenant calling /predictions/severity (US Accidents CC BY-NC-SA 4.0) -> MUST FAIL 403
    r_sev = client.post("/api/v1/predictions/severity", headers=headers, json={
        "latitude": 35.0456,
        "longitude": -85.3097,
        "timezone": "America/New_York",
        "units": "metric",
        "temperature": 22.0
    })
    assert r_sev.status_code == 403
    err_sev = r_sev.json()
    assert err_sev["detail"]["error_code"] == "COMMERCIAL_LICENSING_RESTRICTION"
    assert "CC BY-NC-SA 4.0" in err_sev["detail"]["license_type"]

    # B. Commercial tenant calling /predictions/risk (Chattanooga Research Benchmark) -> MUST FAIL 403
    r_risk = client.post("/api/v1/predictions/risk", headers=headers, json={
        "road": "00I24E",
        "mile": 184.7,
        "hour": 14,
        "speed_mean": 45.0,
        "volume_mean": 20.0,
        "occupancy_mean": 15.0
    })
    assert r_risk.status_code == 403
    err_risk = r_risk.json()
    assert err_risk["detail"]["error_code"] == "COMMERCIAL_LICENSING_RESTRICTION"

    # C. Commercial tenant calling /predictions/combined -> MUST FAIL 403
    r_comb = client.post("/api/v1/predictions/combined", headers=headers, json={
        "risk_inputs": {
            "road": "00I24E", "mile": 184.7, "hour": 14,
            "speed_mean": 45.0, "volume_mean": 20.0, "occupancy_mean": 15.0
        },
        "severity_inputs": {
            "latitude": 35.0456, "longitude": -85.3097, "timezone": "America/New_York"
        }
    })
    assert r_comb.status_code == 403
    err_comb = r_comb.json()
    assert err_comb["detail"]["error_code"] == "COMBINED_ROUTE_LICENSING_RESTRICTION"

    # D. Commercial tenant calling Commercially Cleared Human Injury Severity -> MUST SUCCEED 200
    r_inj = client.post("/api/v1/predictions/injury-severity", headers=headers, json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "units": "metric",
        "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 48.0
    })
    assert r_inj.status_code == 200
    assert "predicted_severity" in r_inj.json()

    # E. Commercial tenant calling Commercially Cleared Road Exposure Risk -> MUST SUCCEED 200
    r_exp = client.post("/api/v1/predictions/exposure-risk", headers=headers, json={
        "road_name": "M1",
        "road_class": "MOTORWAY",
        "link_length_km": 5.0,
        "speed_limit_kmh": 112.7,
        "latitude": 52.1,
        "longitude": -0.7,
        "annual_average_daily_flow": 80000
    })
    assert r_exp.status_code == 200
    assert "predicted_annual_crashes" in r_exp.json()

def test_research_account_can_access_research_models(client: TestClient):
    """Academic / Research accounts with research_access_permitted=True are allowed
    to evaluate experimental benchmarks."""
    reg_res = client.post("/api/v1/auth/register", json={
        "email": "researcher@mit.edu",
        "password": "SecurePassword123!",
        "organization_name": "MIT Transportation Lab",
        "subscription_tier": "FREE",
        "research_access_requested": True
    })
    token = reg_res.json()["tokens"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Research account calling /predictions/severity -> SUCCEEDS
    r_sev = client.post("/api/v1/predictions/severity", headers=headers, json={
        "latitude": 35.0456,
        "longitude": -85.3097,
        "timezone": "America/New_York",
        "units": "metric",
        "temperature": 22.0
    })
    assert r_sev.status_code == 200
    assert "predicted_severity" in r_sev.json()
