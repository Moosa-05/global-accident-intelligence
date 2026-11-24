"""Tests for Multi-Tenant Authentication, User Registration, JWT Tokens, and RBAC."""
import pytest
from fastapi.testclient import TestClient

def test_register_organization_and_user(client: TestClient):
    payload = {
        "email": "owner@fleetcorp.com",
        "password": "SecurePassword123!",
        "full_name": "Fleet Corp Director",
        "organization_name": "Fleet Corp Logistics",
        "subscription_tier": "DEVELOPER"
    }
    r = client.post("/api/v1/auth/register", json=payload)
    assert r.status_code == 201
    data = r.json()
    assert "user" in data
    assert data["user"]["email"] == "owner@fleetcorp.com"
    assert data["user"]["role"] == "OWNER"
    assert data["organization"]["name"] == "Fleet Corp Logistics"
    assert data["organization"]["subscription_tier"] == "DEVELOPER"
    assert data["organization"]["is_commercial_cleared"] is True
    assert "tokens" in data
    assert "access_token" in data["tokens"]

def test_duplicate_email_registration_rejected(client: TestClient):
    payload = {
        "email": "owner@fleetcorp.com",
        "password": "SecurePassword123!",
        "organization_name": "Duplicate Attempt"
    }
    r = client.post("/api/v1/auth/register", json=payload)
    assert r.status_code == 400
    assert "already exists" in r.json()["detail"]

def test_login_and_token_refresh(client: TestClient):
    # 1. Login with correct credentials
    login_res = client.post("/api/v1/auth/login", json={
        "email": "owner@fleetcorp.com",
        "password": "SecurePassword123!"
    })
    assert login_res.status_code == 200
    tokens = login_res.json()
    assert "access_token" in tokens
    assert "refresh_token" in tokens

    # 2. Login with incorrect password
    bad_login = client.post("/api/v1/auth/login", json={
        "email": "owner@fleetcorp.com",
        "password": "WrongPassword!"
    })
    assert bad_login.status_code == 401

    # 3. Refresh access token
    ref_res = client.post("/api/v1/auth/refresh", json={
        "refresh_token": tokens["refresh_token"]
    })
    assert ref_res.status_code == 200
    new_tokens = ref_res.json()
    assert "access_token" in new_tokens

def test_get_current_user_profile(client: TestClient):
    login_res = client.post("/api/v1/auth/login", json={
        "email": "owner@fleetcorp.com",
        "password": "SecurePassword123!"
    })
    token = login_res.json()["access_token"]

    r = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert data["user"]["email"] == "owner@fleetcorp.com"
    assert data["organization"]["name"] == "Fleet Corp Logistics"

def test_unauthenticated_request_rejected_on_protected_routes(client: TestClient):
    r = client.get("/api/v1/auth/me")
    assert r.status_code == 401
