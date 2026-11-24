from fastapi.testclient import TestClient

def test_combined_prediction_endpoint(client: TestClient):
    payload = {
        "risk_inputs": {
            "road": "00I24E",
            "mile": 184.7,
            "hour": 15,
            "weather": "Clear",
            "light": "Daylight",
            "speed_mean": 48.0,
            "speed_std": 5.5,
            "speed_drop": -10.0,
            "volume_mean": 22.0,
            "volume_std": 4.0,
            "occupancy_mean": 18.0,
            "occupancy_max": 28.0,
            "occupancy_std": 4.0,
            "speed_upstream": 55.0,
            "speed_downstream": 40.0
        },
        "severity_inputs": {
            "latitude": 35.0456,
            "longitude": -85.3097,
            "timezone": "America/New_York",
            "units": "metric",
            "temperature": 24.0,
            "humidity": 60.0,
            "pressure": 1013.0,
            "visibility": 10.0,
            "wind_speed": 12.0,
            "precipitation": 0.0,
            "weather_condition": "Clear",
            "is_junction": True,
            "is_traffic_signal": True
        }
    }
    response = client.post("/api/v1/predictions/combined", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    
    # Check risk summary
    risk = data["accident_risk"]
    assert 0.0 <= risk["risk_score"] <= 1.0
    assert risk["risk_level"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    assert isinstance(risk["accident_alert"], bool)
    
    # Check severity summary
    sev = data["conditional_severity"]
    assert sev["predicted_severity"] in [1, 2, 3, 4]
    assert len(sev["severity_probabilities"]) == 4
    assert "Traffic Impact Delay Severity" in sev["definition"]
    
    # Check composite scores
    assert 1.0 <= data["expected_severity_index"] <= 4.0
    assert 0.0 <= data["joint_risk_impact_score"] <= 1.0
    assert "risk_model" in data["model_versions"]
    assert "severity_model" in data["model_versions"]
    assert "geographic_metadata" in data
    assert data["geographic_metadata"]["is_out_of_distribution"] is False
    assert data["geographic_metadata"]["validation_status"] == "EMPIRICALLY_VALIDATED"


def test_combined_prediction_international_ood(client: TestClient):
    payload = {
        "risk_inputs": {
            "road": "M25",
            "mile": 10.0,
            "hour": 8,
            "weather": "Rain",
            "light": "Daylight",
            "speed_mean": 60.0,
            "speed_std": 8.0,
            "speed_drop": -15.0,
            "volume_mean": 30.0,
            "volume_std": 5.0,
            "occupancy_mean": 25.0,
            "occupancy_max": 35.0,
            "occupancy_std": 5.0,
            "speed_upstream": 70.0,
            "speed_downstream": 50.0
        },
        "severity_inputs": {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "timezone": "Europe/Paris",
            "units": "metric",
            "temperature": 10.0,
            "humidity": 80.0,
            "pressure": 1015.0,
            "visibility": 7.0,
            "wind_speed": 20.0,
            "precipitation": 2.5,
            "weather_condition": "Rain",
            "is_junction": True
        }
    }
    response = client.post("/api/v1/predictions/combined", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert data["geographic_metadata"]["is_out_of_distribution"] is True
    assert data["geographic_metadata"]["validation_status"] == "UNVALIDATED_TERRITORY"
    assert "outside the empirical training coverage" in data["geographic_metadata"]["validation_warning"]

