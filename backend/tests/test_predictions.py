from fastapi.testclient import TestClient

def test_risk_prediction_valid_input(client: TestClient):
    payload = {
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
    response = client.post("/api/v1/predictions/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_level"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert isinstance(data["accident_alert"], bool)
    assert data["threshold_used"] > 0
    assert data["model_version"].startswith("1.")
    assert data["latency_ms"] >= 0

def test_risk_prediction_high_congestion(client: TestClient):
    payload = {
        "road": "00I75S",
        "mile": 4.3,
        "hour": 17,
        "weather": "Rain",
        "light": "Dusk",
        "speed_mean": 15.0,
        "speed_std": 8.5,
        "speed_drop": -25.0,
        "volume_mean": 28.0,
        "volume_std": 6.0,
        "occupancy_mean": 35.0,
        "occupancy_max": 65.0,
        "occupancy_std": 12.0,
        "speed_upstream": 45.0,
        "speed_downstream": 10.0
    }
    response = client.post("/api/v1/predictions/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0.0 <= data["risk_score"] <= 1.0

def test_risk_prediction_missing_required_fields(client: TestClient):
    # Missing speed_mean, volume_mean, occupancy_mean
    payload = {
        "road": "00I24E",
        "mile": 184.7
    }
    response = client.post("/api/v1/predictions/risk", json=payload)
    assert response.status_code == 422
