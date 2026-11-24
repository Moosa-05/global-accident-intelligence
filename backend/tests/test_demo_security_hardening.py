"""Tests for Demonstration Security Hardening:
1. Cryptographic session token generation & verification.
2. Tampered session token rejection.
3. Prediction history session isolation (preventing cross-visitor data leakage).
4. Unauthenticated execution of all four prediction pipelines.
5. IP-based rate limiting on public endpoints.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.rate_limiter import rate_limiter
from backend.app.api.deps import generate_signed_session_token, verify_session_token

client = TestClient(app)

def test_session_token_cryptographic_verification():
    """Verify HMAC generation and tampering detection."""
    session_id = "test_sess_12345"
    token = generate_signed_session_token(session_id, settings.SECRET_KEY)
    
    # 1. Valid token returns original session_id
    verified = verify_session_token(token, settings.SECRET_KEY)
    assert verified == session_id

    # 2. Tampered session_id is rejected
    tampered_id_token = f"tampered_{token}"
    assert verify_session_token(tampered_id_token, settings.SECRET_KEY) is None

    # 3. Tampered signature is rejected
    uuid_part, sig_part = token.split(".")
    tampered_sig_token = f"{uuid_part}.{'a' * len(sig_part)}"
    assert verify_session_token(tampered_sig_token, settings.SECRET_KEY) is None

    # 4. Empty or malformed token is rejected
    assert verify_session_token("", settings.SECRET_KEY) is None
    assert verify_session_token("just_a_string_no_dot", settings.SECRET_KEY) is None


def test_anonymous_prediction_history_session_isolation():
    """Verify that Visitor A and Visitor B have complete isolation of prediction history."""
    # Visitor A establishes session
    r1 = client.get("/api/v1/models")
    assert r1.status_code == 200
    token_a = r1.headers.get("X-Session-ID")
    assert token_a is not None
    assert verify_session_token(token_a, settings.SECRET_KEY) is not None

    # Visitor A performs an injury severity prediction
    injury_payload = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "speed_limit": 30,
        "number_of_vehicles": 2,
        "number_of_casualties": 1,
        "light_conditions": "DAYLIGHT",
        "weather_conditions": "FINE_NO_HIGH_WINDS",
        "road_surface_conditions": "DRY",
        "vehicle_type": "CAR",
        "age_of_driver": 35,
        "sex_of_driver": "MALE",
        "age_of_vehicle": 4,
        "engine_capacity_cc": 1600,
        "propulsion_code": "PETROL"
    }
    r_pred_a = client.post("/api/v1/predictions/injury-severity", json=injury_payload, headers={"X-Session-ID": token_a})
    assert r_pred_a.status_code == 200
    pred_data_a = r_pred_a.json()
    assert "predicted_severity" in pred_data_a

    # Visitor A queries history: must see their prediction
    r_hist_a = client.get("/api/v1/history/predictions", headers={"X-Session-ID": token_a})
    assert r_hist_a.status_code == 200
    hist_a = r_hist_a.json()
    assert hist_a["total"] >= 1
    found_a = any(rec["id"] == pred_data_a["prediction_id"] for rec in hist_a["records"])
    assert found_a is True

    # Visitor B arrives with a brand new independent session
    token_b = generate_signed_session_token("session_visitor_b", settings.SECRET_KEY)
    r_hist_b = client.get("/api/v1/history/predictions", headers={"X-Session-ID": token_b})
    assert r_hist_b.status_code == 200
    hist_b = r_hist_b.json()
    
    # Visitor B MUST NOT see Visitor A's prediction
    found_b = any(rec["id"] == pred_data_a["prediction_id"] for rec in hist_b["records"])
    assert found_b is False


def test_tampered_session_token_rejected_and_reissued():
    """Verify that an untrusted or tampered client token is replaced with a new secure token."""
    # Client sends an arbitrary forged token
    forged_token = "admin_attacker.fake_signature_12345"
    r = client.get("/api/v1/history/predictions", headers={"X-Session-ID": forged_token})
    assert r.status_code == 200
    
    # The response must issue a newly generated, validly signed token
    reissued_token = r.headers.get("X-Session-ID")
    assert reissued_token is not None
    assert reissued_token != forged_token
    assert verify_session_token(reissued_token, settings.SECRET_KEY) is not None


def test_all_four_prediction_pipelines_work_without_login():
    """Verify that all four prediction endpoints execute successfully in demonstration mode."""
    # 1. Human Injury Severity (UK/France)
    injury_payload = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "speed_limit": 30,
        "number_of_vehicles": 1,
        "number_of_casualties": 1,
        "light_conditions": "DAYLIGHT",
        "weather_conditions": "FINE_NO_HIGH_WINDS",
        "road_surface_conditions": "DRY",
        "vehicle_type": "CAR",
        "age_of_driver": 28,
        "sex_of_driver": "FEMALE",
        "age_of_vehicle": 2,
        "engine_capacity_cc": 1400,
        "propulsion_code": "PETROL"
    }
    r1 = client.post("/api/v1/predictions/injury-severity", json=injury_payload)
    assert r1.status_code == 200
    assert "predicted_severity" in r1.json()

    # 2. Exposure Risk (UK DfT)
    exposure_payload = {
        "road_name": "M1",
        "road_class": "MOTORWAY",
        "link_length_km": 5.0,
        "speed_limit_kmh": 110.0,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "annual_average_daily_flow": 45000
    }
    r2 = client.post("/api/v1/predictions/exposure-risk", json=exposure_payload)
    assert r2.status_code == 200
    assert "predicted_annual_crashes" in r2.json()

    # 3. Corridor Risk (15-min Operational)
    corridor_payload = {
        "road": "00I24E",
        "mile": 184.7,
        "hour": 14,
        "weather": "Clear",
        "light": "Daylight",
        "speed_mean": 62.5,
        "speed_std": 3.1,
        "speed_drop": -2.0,
        "volume_mean": 14.0,
        "volume_std": 2.5,
        "occupancy_mean": 4.2,
        "occupancy_max": 8.0,
        "occupancy_std": 1.1,
        "speed_upstream": 60.0,
        "volume_upstream": 12.0,
        "occupancy_upstream": 3.8,
        "speed_downstream": 64.0,
        "volume_downstream": 15.0,
        "occupancy_downstream": 4.5
    }
    r3 = client.post("/api/v1/predictions/risk", json=corridor_payload)
    assert r3.status_code == 200
    assert "risk_score" in r3.json()

    # 4. Traffic Severity (Research Model)
    severity_payload = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "temperature": 18.0,
        "humidity": 60.0,
        "pressure": 1013.25,
        "visibility": 10.0,
        "wind_speed": 12.0,
        "precipitation": 0.0,
        "hour_of_day": 17,
        "day_of_week": 4,
        "month": 10,
        "distance": 0.5,
        "side": "R",
        "crossing": False,
        "junction": True,
        "traffic_signal": True,
        "sunrise_sunset": "Day",
        "weather_condition": "Clear"
    }
    r4 = client.post("/api/v1/predictions/severity", json=severity_payload)
    assert r4.status_code == 200
    assert "predicted_severity" in r4.json()


def test_ip_based_rate_limiting_enforcement():
    """Verify that demonstration IP rate limit (60 req/min) rejects excess requests with 429."""
    rate_limiter.reset("demo_ip_192.0.2.1")
    headers = {"X-Forwarded-For": "192.0.2.1"}

    # Simulate 60 allowed requests
    for _ in range(60):
        allowed, _, _ = rate_limiter.is_allowed("demo_ip_192.0.2.1", 60)
        assert allowed is True

    # 61st request from same IP must be rejected with 429
    r_blocked = client.get("/api/v1/models", headers=headers)
    assert r_blocked.status_code == 429
    assert "Rate limit exceeded" in r_blocked.json()["detail"]
    assert "Retry-After" in r_blocked.headers

    rate_limiter.reset("demo_ip_192.0.2.1")
