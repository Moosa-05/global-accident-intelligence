from fastapi.testclient import TestClient

def test_severity_prediction_global_coordinates_metric(client: TestClient):
    # Test request for London, UK (51.5074 N, 0.1278 W)
    payload = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timezone": "Europe/London",
        "units": "metric",
        "temperature": 12.0,
        "humidity": 75.0,
        "pressure": 1012.0,
        "visibility": 8.0,
        "wind_speed": 15.0,
        "precipitation": 1.2,
        "weather_condition": "Light Rain",
        "is_junction": True,
        "is_traffic_signal": True
    }
    response = client.post("/api/v1/predictions/severity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_id" in data
    assert data["predicted_severity"] in [1, 2, 3, 4]
    assert data["severity_level"] in ["MINIMAL_IMPACT", "MODERATE_IMPACT", "SIGNIFICANT_IMPACT", "SEVERE_DISRUPTION"]
    
    probs = data["severity_probabilities"]
    assert "class_1" in probs and "class_2" in probs and "class_3" in probs and "class_4" in probs
    total_prob = sum(probs.values())
    assert 0.99 <= total_prob <= 1.01
    
    assert data["model_version"].startswith("2.")
    assert "geographic_metadata" in data
    assert data["geographic_metadata"]["coordinates"]["latitude"] == 51.5074
    # London is outside the US training footprint -> Out-of-Distribution warning flagged
    assert data["geographic_metadata"]["is_out_of_distribution"] is True
    assert data["geographic_metadata"]["validation_status"] == "UNVALIDATED_TERRITORY"
    assert data["geographic_metadata"]["validation_warning"] is not None
    assert "extrapolation" in data["geographic_metadata"]["validation_warning"].lower() or "outside" in data["geographic_metadata"]["validation_warning"].lower()

def test_severity_prediction_imperial_units(client: TestClient):
    payload = {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "timezone": "America/Los_Angeles",
        "units": "imperial",
        "temperature": 68.0,
        "humidity": 45.0,
        "pressure": 29.92,
        "visibility": 10.0,
        "wind_speed": 8.0,
        "precipitation": 0.0,
        "weather_condition": "Clear",
        "is_crossing": True
    }
    response = client.post("/api/v1/predictions/severity", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_severity"] in [1, 2, 3, 4]
    # Los Angeles is inside the validated training geography
    assert data["geographic_metadata"]["is_out_of_distribution"] is False
    assert data["geographic_metadata"]["validation_status"] == "EMPIRICALLY_VALIDATED"
    assert data["geographic_metadata"]["validation_warning"] is None


def test_severity_prediction_invalid_latitude(client: TestClient):
    payload = {
        "latitude": 95.0,  # Invalid latitude (> 90)
        "longitude": 0.0
    }
    response = client.post("/api/v1/predictions/severity", json=payload)
    assert response.status_code == 422
