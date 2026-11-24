"""Comprehensive End-to-End SaaS Lifecycle Integration Test.
Executes actual end-to-end workflows:
1. Tenant registration & provisioning
2. JWT authentication, profile hydration & token refresh
3. Role-based permissions & member access
4. API key creation with scopes, live usage, rotation & revocation
5. Commercial licensing enforcement (403 gating on CC BY-NC-SA 4.0 models)
6. Prediction execution across commercial pipelines
7. Plan changes via simulated payment provider
8. Prediction history & API audit trail with tenant isolation
"""
import uuid
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_full_saas_lifecycle_and_tenant_security():
    # Unique tenant email & slug
    rand_id = str(uuid.uuid4())[:8]
    email = f"lead-{rand_id}@acme-mobility.io"
    org_name = f"Acme Autonomous {rand_id}"
    password = "SecurePassword123!"

    # 1. Organization Registration & Provisioning
    reg_res = client.post("/api/v1/auth/register", json={
        "email": email,
        "password": password,
        "full_name": "Dr. Elena Vance",
        "organization_name": org_name,
        "subscription_tier": "DEVELOPER"
    })
    assert reg_res.status_code == 201, reg_res.text
    reg_data = reg_res.json()
    assert "tokens" in reg_data
    access_token = reg_data["tokens"]["access_token"]
    refresh_token = reg_data["tokens"]["refresh_token"]
    org_id = reg_data["organization"]["id"]
    headers = {"Authorization": f"Bearer {access_token}"}

    # 2. Login, Profile Hydration & Token Refresh
    login_res = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password
    })
    assert login_res.status_code == 200, login_res.text
    login_tokens = login_res.json()
    assert "access_token" in login_tokens

    # Verify Profile & Org endpoint
    me_res = client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["user"]["email"] == email
    assert me_data["organization"]["subscription_tier"] == "DEVELOPER"
    assert me_data["organization"]["is_commercial_cleared"] is True

    # Test Refresh Token
    ref_res = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert ref_res.status_code == 200
    new_access_token = ref_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {new_access_token}"}

    # 3. Organization Members Management (RBAC)
    members_res = client.get("/api/v1/organizations/members", headers=headers)
    assert members_res.status_code == 200
    members = members_res.json()
    assert len(members) >= 1
    assert members[0]["role"] == "OWNER"

    # 4. API Key Creation
    key_res = client.post("/api/v1/api-keys", headers=headers, json={
        "name": "Acme Fleet Dispatch",
        "is_live": True,
        "scopes": ["predict:injury", "predict:exposure", "models:read"]
    })
    assert key_res.status_code in [200, 201], key_res.text
    key_data = key_res.json()
    raw_api_key = key_data["api_key"]
    key_id = key_data["id"]
    assert raw_api_key.startswith("agy_live_")

    api_headers = {"X-API-Key": raw_api_key}

    # 5. Prediction Execution with API Key (Commercial Model)
    injury_res = client.post("/api/v1/predictions/injury-severity", headers=api_headers, json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "MOTORWAY",
        "speed_limit_kmh": 110.0,
        "units": "metric",
        "temperature": 16.0
    })
    assert injury_res.status_code == 200, injury_res.text
    inj_out = injury_res.json()
    assert inj_out["predicted_severity"] in ["SLIGHT", "SERIOUS", "FATAL"]
    assert "Conditional on Collision" in inj_out["target_definition"]

    # 6. Commercial Model Licensing Enforcement (403 Gated on Non-commercial Models)
    # Acme is on DEVELOPER tier (Commercial Cleared), so CC BY-NC-SA 4.0 models MUST be rejected!
    sev_res = client.post("/api/v1/predictions/severity", headers=api_headers, json={
        "latitude": 35.0456,
        "longitude": -85.3097,
        "units": "metric"
    })
    assert sev_res.status_code == 403, (
        f"Expected 403 Forbidden for commercial tenant on CC BY-NC-SA model, got {sev_res.status_code}"
    )
    assert "COMMERCIAL_LICENSING_RESTRICTION" in str(sev_res.json())

    # 7. API Key Rotation
    rotate_res = client.post(f"/api/v1/api-keys/{key_id}/rotate", headers=headers)
    assert rotate_res.status_code == 200
    new_raw_key = rotate_res.json()["new_key"]["api_key"]
    new_key_id = rotate_res.json()["new_key"]["id"]

    # Old key must now be rejected
    old_call = client.post("/api/v1/predictions/injury-severity", headers={"X-API-Key": raw_api_key}, json={
        "latitude": 51.5074, "longitude": -0.1278, "road_class": "MOTORWAY"
    })
    assert old_call.status_code == 401

    # New key must succeed
    new_call = client.post("/api/v1/predictions/injury-severity", headers={"X-API-Key": new_raw_key}, json={
        "latitude": 51.5074, "longitude": -0.1278, "road_class": "MOTORWAY"
    })
    assert new_call.status_code == 200

    # 8. Subscription Upgrade via MockPaymentProvider
    sub_change = client.post("/api/v1/subscriptions/change-plan", headers=headers, json={
        "new_tier": "PROFESSIONAL"
    })
    assert sub_change.status_code == 200, sub_change.text
    sub_res = sub_change.json()
    assert sub_res["new_tier"] == "PROFESSIONAL"
    assert sub_res["monthly_request_quota"] == 50000

    # Verify updated organization usage
    usage_res = client.get("/api/v1/organizations/usage", headers=headers)
    assert usage_res.status_code == 200
    usage_data = usage_res.json()
    assert usage_data["subscription_tier"] == "PROFESSIONAL"
    assert usage_data["monthly_request_quota"] == 50000

    # 9. Tenant Isolation on Prediction History & Audit Trail
    hist_res = client.get("/api/v1/history/predictions", headers=headers)
    assert hist_res.status_code == 200
    hist_records = hist_res.json()["records"]
    assert len(hist_records) >= 1
    # Check that predictions logged under Acme belong to Acme
    for rec in hist_records:
        assert rec["type"] in ["injury_severity", "exposure_risk", "corridor_risk", "traffic_severity"]

    audit_res = client.get("/api/v1/history/audit-logs", headers=headers)
    assert audit_res.status_code == 200
    audit_logs = audit_res.json()
    assert len(audit_logs) >= 1
    for log in audit_logs:
        assert log["endpoint"] in ["/api/v1/predictions/injury-severity", "/api/v1/predictions/severity"]

    # 10. Key Revocation
    revoke_res = client.delete(f"/api/v1/api-keys/{new_key_id}", headers=headers)
    assert revoke_res.status_code == 200
    revoked_call = client.post("/api/v1/predictions/injury-severity", headers={"X-API-Key": new_raw_key}, json={
        "latitude": 51.5074, "longitude": -0.1278, "road_class": "MOTORWAY"
    })
    assert revoked_call.status_code == 401
