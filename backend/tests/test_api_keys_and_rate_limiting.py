"""Tests for Scoped API Keys, Key Rotation, Revocation, and Sliding-Window Rate Limiting."""
import pytest
from fastapi.testclient import TestClient
from backend.app.core.rate_limiter import rate_limiter

@pytest.fixture
def auth_headers(client: TestClient):
    # Register a new tenant for API key tests
    client.post("/api/v1/auth/register", json={
        "email": "admin@telematics.io",
        "password": "SecurePassword123!",
        "organization_name": "Telematics Fleet IO",
        "subscription_tier": "DEVELOPER"
    })
    login_res = client.post("/api/v1/auth/login", json={
        "email": "admin@telematics.io",
        "password": "SecurePassword123!"
    })
    token = login_res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

def test_create_and_list_api_keys(client: TestClient, auth_headers):
    # 1. Create API Key
    r = client.post("/api/v1/api-keys", headers=auth_headers, json={
        "name": "Production Road Sensor Feed",
        "scopes": ["predict:exposure", "predict:injury"]
    })
    assert r.status_code == 201
    key_data = r.json()
    assert "api_key" in key_data
    assert key_data["api_key"].startswith("agy_live_")
    assert key_data["name"] == "Production Road Sensor Feed"
    api_key_str = key_data["api_key"]
    key_id = key_data["id"]

    # 2. List API Keys (should hide secret, showing only prefix)
    list_res = client.get("/api/v1/api-keys", headers=auth_headers)
    assert list_res.status_code == 200
    keys = list_res.json()
    assert len(keys) >= 1
    found = next(k for k in keys if k["id"] == key_id)
    assert "api_key" not in found  # Secret must NOT be returned in list
    assert found["key_prefix"].startswith("agy_live_")

    # 3. Use API Key to invoke commercial injury severity endpoint
    pred_res = client.post("/api/v1/predictions/injury-severity", headers={"X-API-Key": api_key_str}, json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "units": "metric",
        "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 48.0
    })
    assert pred_res.status_code == 200
    assert "predicted_severity" in pred_res.json()

def test_rotate_and_revoke_api_key(client: TestClient, auth_headers):
    # 1. Create Key
    r = client.post("/api/v1/api-keys", headers=auth_headers, json={"name": "Rotatable Key"})
    key_data = r.json()
    key_id = key_data["id"]
    old_key_str = key_data["api_key"]

    # 2. Rotate Key
    rot_res = client.post(f"/api/v1/api-keys/{key_id}/rotate", headers=auth_headers)
    assert rot_res.status_code == 200
    rot_data = rot_res.json()
    assert rot_data["old_key_id"] == key_id
    new_key_str = rot_data["new_key"]["api_key"]

    # 3. Old key must now be rejected
    old_use = client.post("/api/v1/predictions/exposure-risk", headers={"X-API-Key": old_key_str}, json={
        "road_name": "M1",
        "road_class": "MOTORWAY",
        "link_length_km": 5.0,
        "speed_limit_kmh": 112.7,
        "latitude": 52.1,
        "longitude": -0.7,
        "annual_average_daily_flow": 80000
    })
    assert old_use.status_code == 401

    # 4. New key must succeed
    new_use = client.post("/api/v1/predictions/exposure-risk", headers={"X-API-Key": new_key_str}, json={
        "road_name": "M1",
        "road_class": "MOTORWAY",
        "link_length_km": 5.0,
        "speed_limit_kmh": 112.7,
        "latitude": 52.1,
        "longitude": -0.7,
        "annual_average_daily_flow": 80000
    })
    assert new_use.status_code == 200

    # 5. Revoke new key
    new_key_id = rot_data["new_key"]["id"]
    rev_res = client.delete(f"/api/v1/api-keys/{new_key_id}", headers=auth_headers)
    assert rev_res.status_code == 200

    # 6. Revoked key rejected
    rev_use = client.post("/api/v1/predictions/exposure-risk", headers={"X-API-Key": new_key_str}, json={
        "road_name": "M1",
        "road_class": "MOTORWAY",
        "link_length_km": 5.0,
        "speed_limit_kmh": 112.7,
        "latitude": 52.1,
        "longitude": -0.7,
        "annual_average_daily_flow": 80000
    })
    assert rev_use.status_code == 401

def test_rate_limiter_enforcement(client: TestClient):
    rate_limiter.reset()
    # Test rate limit directly with in-memory limiter
    for _ in range(10):
        allowed, count, retry = rate_limiter.is_allowed("test_tenant_key", max_requests_per_minute=10)
        assert allowed is True
    
    # 11th request must be rejected
    allowed, count, retry = rate_limiter.is_allowed("test_tenant_key", max_requests_per_minute=10)
    assert allowed is False
    assert retry > 0
    rate_limiter.reset()
