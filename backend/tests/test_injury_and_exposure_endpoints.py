"""Integration tests for Human Injury Severity and Exposure Risk endpoints."""
from fastapi.testclient import TestClient

def test_injury_severity_prediction_uk(client: TestClient):
    payload = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timezone": "Europe/London",
        "units": "metric",
        "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 48.0,
        "temperature": 14.0,
        "humidity": 75.0,
        "pressure": 1012.0,
        "wind_speed": 18.0,
        "precipitation": 0.5,
        "weather_condition": "LIGHT_RAIN",
        "is_junction": True,
        "is_traffic_signal": True
    }
    response = client.post("/api/v1/predictions/injury-severity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert data["predicted_severity"] in ["SLIGHT", "SERIOUS", "FATAL"]
    assert data["severity_level_code"] in [0, 1, 2]
    
    probs = data["probabilities"]
    assert "SLIGHT" in probs and "SERIOUS" in probs and "FATAL" in probs
    total_prob = sum(probs.values())
    assert 0.99 <= total_prob <= 1.01
    
    assert isinstance(data["fatality_risk_alert"], bool)
    assert data["model_version"] == "1.0.0"
    assert "geographic_metadata" in data
    assert data["geographic_metadata"]["validation_status"] == "EMPIRICALLY_VALIDATED"

def test_injury_severity_prediction_france_imperial(client: TestClient):
    payload = {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "timezone": "Europe/Paris",
        "units": "imperial",
        "road_class": "MOTORWAY",
        "speed_limit_kmh": 110.0,
        "temperature": 60.0,
        "humidity": 60.0,
        "pressure": 29.92,
        "wind_speed": 12.0,
        "precipitation": 0.0,
        "weather_condition": "CLEAR",
        "is_junction": False
    }
    response = client.post("/api/v1/predictions/injury-severity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_severity"] in ["SLIGHT", "SERIOUS", "FATAL"]
    assert data["geographic_metadata"]["validation_status"] == "EMPIRICALLY_VALIDATED"

def test_injury_severity_prediction_ood(client: TestClient):
    # Tokyo, Japan
    payload = {
        "latitude": 35.6762,
        "longitude": 139.6503,
        "timezone": "Asia/Tokyo",
        "units": "metric",
        "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 50.0
    }
    response = client.post("/api/v1/predictions/injury-severity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["geographic_metadata"]["is_out_of_distribution"] is True
    assert data["geographic_metadata"]["validation_status"] == "UNVALIDATED_TERRITORY"

def test_exposure_risk_prediction_motorway(client: TestClient):
    payload = {
        "road_name": "M1",
        "road_class": "MOTORWAY",
        "link_length_km": 5.2,
        "speed_limit_kmh": 112.7,
        "latitude": 52.1386,
        "longitude": -0.7301,
        "annual_average_daily_flow": 75000
    }
    response = client.post("/api/v1/predictions/exposure-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert data["annual_exposure_million_vkt"] > 0
    assert data["predicted_annual_crashes"] >= 0
    assert data["predicted_crash_rate_per_million_vkt"] >= 0
    assert data["risk_tier"] in ["LOW_RISK", "TYPICAL_RISK", "ELEVATED_RISK", "HIGH_RISK"]
    assert "Negative Binomial GLM" in data["statistical_model"]

def test_exposure_risk_prediction_invalid_length(client: TestClient):
    payload = {
        "road_name": "M1",
        "road_class": "MOTORWAY",
        "link_length_km": -2.0,  # Invalid negative length
        "speed_limit_kmh": 100.0,
        "latitude": 52.0,
        "longitude": -0.5,
        "annual_average_daily_flow": 50000
    }
    response = client.post("/api/v1/predictions/exposure-risk", json=payload)
    assert response.status_code == 422
