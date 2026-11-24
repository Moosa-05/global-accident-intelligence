"""Automated Global Intelligence & Multi-Country Verification Tests.
Verifies:
- Worldwide coordinate acceptance (WGS84 EPSG:4326)
- Imperial vs Metric unit conversion consistency
- IANA timezone handling
- Empirical validation bounds (UK/France/US) vs Out-of-Distribution (OOD) detection
- Tested across London, Paris, New York, Tokyo, Sydney
"""
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

LOCATIONS = [
    {
        "name": "London",
        "lat": 51.5074,
        "lng": -0.1278,
        "timezone": "Europe/London",
        "expected_injury_ood": False,
        "expected_severity_ood": True,  # US severity model is OOD in Europe
    },
    {
        "name": "Paris",
        "lat": 48.8566,
        "lng": 2.3522,
        "timezone": "Europe/Paris",
        "expected_injury_ood": False,
        "expected_severity_ood": True,
    },
    {
        "name": "New York",
        "lat": 40.7128,
        "lng": -74.0060,
        "timezone": "America/New_York",
        "expected_injury_ood": True,   # Injury model is UK/FR validated
        "expected_severity_ood": False, # US delay model is validated
    },
    {
        "name": "Tokyo",
        "lat": 35.6762,
        "lng": 139.6503,
        "timezone": "Asia/Tokyo",
        "expected_injury_ood": True,
        "expected_severity_ood": True,
    },
    {
        "name": "Sydney",
        "lat": -33.8688,
        "lng": 151.2093,
        "timezone": "Australia/Sydney",
        "expected_injury_ood": True,
        "expected_severity_ood": True,
    },
]

def test_injury_severity_global_locations():
    """Verifies that human injury severity endpoint accepts all worldwide locations,
    evaluates OOD flags properly, and formats conditional-on-collision definitions.
    """
    for loc in LOCATIONS:
        payload = {
            "latitude": loc["lat"],
            "longitude": loc["lng"],
            "timezone": loc["timezone"],
            "road_class": "PRIMARY_ARTERIAL",
            "speed_limit_kmh": 60.0,
            "units": "metric",
            "temperature": 18.0,
            "wind_speed": 12.0
        }
        res = client.post("/api/v1/predictions/injury-severity", json=payload)
        assert res.status_code == 200, f"Failed for {loc['name']}: {res.text}"
        data = res.json()

        assert data["predicted_severity"] in ["SLIGHT", "SERIOUS", "FATAL"]
        assert "probabilities" in data
        assert "target_definition" in data
        assert "Conditional on Collision" in data["target_definition"]

        geo = data.get("geographic_metadata")
        assert geo is not None, f"Missing geo metadata for {loc['name']}"
        assert geo["is_out_of_distribution"] == loc["expected_injury_ood"], (
            f"OOD mismatch for {loc['name']}: expected {loc['expected_injury_ood']}, got {geo['is_out_of_distribution']}"
        )
        if loc["expected_injury_ood"]:
            assert geo["validation_status"] == "UNVALIDATED_TERRITORY"
            assert geo["validation_warning"] is not None
        else:
            assert geo["validation_status"] == "EMPIRICALLY_VALIDATED"
            assert geo["validation_warning"] is None

def test_exposure_risk_global_locations():
    """Verifies annual exposure risk model accepts global coordinates and computes
    normalized crash rates and risk tiers mathematically.
    """
    for loc in LOCATIONS:
        payload = {
            "road_name": f"{loc['name']} Outer Ring",
            "road_class": "MOTORWAY",
            "link_length_km": 3.2,
            "speed_limit_kmh": 100.0,
            "latitude": loc["lat"],
            "longitude": loc["lng"],
            "annual_average_daily_flow": 42000
        }
        res = client.post("/api/v1/predictions/exposure-risk", json=payload)
        assert res.status_code == 200, f"Failed for {loc['name']}: {res.text}"
        data = res.json()

        assert data["predicted_annual_crashes"] > 0
        assert data["annual_exposure_million_vkt"] > 0
        assert data["predicted_crash_rate_per_million_vkt"] > 0
        assert data["risk_tier"] in ["LOW_RISK", "TYPICAL_RISK", "ELEVATED_RISK", "HIGH_RISK"]

def test_unit_conversion_equiv():
    """Verifies that Metric and Imperial inputs yield consistent predictions."""
    # Metric: 20°C, 24.14 km/h (15 mph), 1013.25 hPa
    res_metric = client.post("/api/v1/predictions/injury-severity", json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "MOTORWAY",
        "speed_limit_kmh": 112.65,
        "units": "metric",
        "temperature": 20.0,
        "wind_speed": 24.14,
        "pressure": 1013.25,
        "precipitation": 0.0
    })
    # Imperial: 68°F, 15 mph, 29.92 inHg
    res_imperial = client.post("/api/v1/predictions/injury-severity", json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "MOTORWAY",
        "speed_limit_kmh": 112.65,
        "units": "imperial",
        "temperature": 68.0,
        "wind_speed": 15.0,
        "pressure": 29.92,
        "precipitation": 0.0
    })
    assert res_metric.status_code == 200
    assert res_imperial.status_code == 200
    data_m = res_metric.json()
    data_i = res_imperial.json()

    # Probabilities should be virtually identical
    assert data_m["predicted_severity"] == data_i["predicted_severity"]
    for k in ["SLIGHT", "SERIOUS", "FATAL"]:
        assert abs(data_m["probabilities"][k] - data_i["probabilities"][k]) < 0.01
