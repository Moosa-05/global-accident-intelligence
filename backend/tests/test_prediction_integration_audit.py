"""Integration & Diagnostic Regression Tests for All Four Prediction Pipelines.
Verifies that prediction outputs respond dynamically to varying input conditions,
aliases and categorical encodings are correctly handled, and no hardcoded/stale outputs persist.
"""
import pytest
from starlette.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_injury_severity_dynamic_response_and_alias_handling():
    """Verify Human Injury Severity produces distinct probabilities across conditions and handles aliases."""
    # Scenario A: Baseline Daylight Motorway Dry 110 km/h
    res_a = client.post("/api/v1/predictions/injury-severity", json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "MOTORWAY",
        "speed_limit_kmh": 110,
        "light_condition": "DAYLIGHT",
        "weather_condition": "CLEAR",
        "road_surface_condition": "DRY",
        "divided_roadway": True
    })
    assert res_a.status_code == 200
    data_a = res_a.json()
    probs_a = data_a["probabilities"]

    # Scenario B: High-Hazard Nighttime Dark Unlit Arterial in Heavy Rain with Junction 120 km/h
    res_b = client.post("/api/v1/predictions/injury-severity", json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 120,
        "light_condition": "DARKNESS_NO_STREETLIGHTS",
        "is_night": True,
        "weather_condition": "HEAVY_RAIN",
        "road_surface_condition": "WET_DAMP",
        "is_junction": True
    })
    assert res_b.status_code == 200
    data_b = res_b.json()
    probs_b = data_b["probabilities"]

    # Scenario C: Low-Speed Urban Residential Street 30 km/h
    res_c = client.post("/api/v1/predictions/injury-severity", json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "LOCAL_RESIDENTIAL",
        "speed_limit_kmh": 30,
        "light_condition": "DAYLIGHT",
        "weather_condition": "CLEAR",
        "road_surface_condition": "DRY",
        "is_junction": False
    })
    assert res_c.status_code == 200
    data_c = res_c.json()
    probs_c = data_c["probabilities"]

    # Verify that outputs are NOT unchanged
    assert probs_a["SLIGHT"] != probs_b["SLIGHT"]
    assert probs_b["FATAL"] > probs_c["FATAL"]  # High speed unlit rain has higher fatality probability
    assert probs_c["SLIGHT"] > probs_b["SLIGHT"]  # Urban low speed has higher slight probability

    # Scenario D: Verify Legacy / Frontend Aliases (A_ROAD, light_conditions, surface_conditions)
    res_alias = client.post("/api/v1/predictions/injury-severity", json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "A_ROAD",  # Alias for PRIMARY_ARTERIAL
        "speed_limit_kmh": 100,
        "light_conditions": "DARK_NO_LIGHTS",  # Alias for DARKNESS_NO_STREETLIGHTS
        "surface_conditions": "WET_OR_DAMP",   # Alias for WET_DAMP
        "weather_conditions": "FINE_NO_HIGH_WINDS"  # Alias for CLEAR
    })
    assert res_alias.status_code == 200
    data_alias = res_alias.json()
    # Ensure aliases were converted and did not collapse to default 50.5%
    assert data_alias["probabilities"]["SLIGHT"] != 0.5048

def test_annual_exposure_risk_dynamic_response_and_math_consistency():
    """Verify Annual Exposure Risk responds dynamically to AADF, length, and road class, with exact math."""
    # Scenario 1: Low-Volume Short Link (1 km, 5,000 AADF)
    res_1 = client.post("/api/v1/predictions/exposure-risk", json={
        "road_name": "B100",
        "road_class": "SECONDARY_COLLECTOR",
        "link_length_km": 1.0,
        "speed_limit_kmh": 60,
        "latitude": 53.48,
        "longitude": -2.24,
        "annual_average_daily_flow": 5000
    })
    assert res_1.status_code == 200
    data_1 = res_1.json()

    # Scenario 2: Medium Motorway Link (2.5 km, 48,500 AADF)
    res_2 = client.post("/api/v1/predictions/exposure-risk", json={
        "road_name": "M62",
        "road_class": "MOTORWAY",
        "link_length_km": 2.5,
        "speed_limit_kmh": 110,
        "latitude": 53.48,
        "longitude": -2.24,
        "annual_average_daily_flow": 48500
    })
    assert res_2.status_code == 200
    data_2 = res_2.json()

    # Scenario 3: High-Volume Long Corridor (12 km, 95,000 AADF)
    res_3 = client.post("/api/v1/predictions/exposure-risk", json={
        "road_name": "M25",
        "road_class": "MOTORWAY",
        "link_length_km": 12.0,
        "speed_limit_kmh": 110,
        "latitude": 51.50,
        "longitude": -0.10,
        "annual_average_daily_flow": 95000
    })
    assert res_3.status_code == 200
    data_3 = res_3.json()

    # Verify mathematical consistency: MVKT = AADF * link_length_km * 365 / 1e6
    expected_mvkt_1 = round(5000 * 1.0 * 365 / 1e6, 4)
    expected_mvkt_2 = round(48500 * 2.5 * 365 / 1e6, 4)
    expected_mvkt_3 = round(95000 * 12.0 * 365 / 1e6, 4)

    assert abs(data_1["annual_exposure_million_vkt"] - expected_mvkt_1) < 1e-3
    assert abs(data_2["annual_exposure_million_vkt"] - expected_mvkt_2) < 1e-3
    assert abs(data_3["annual_exposure_million_vkt"] - expected_mvkt_3) < 1e-3

    # Expected crash count must scale with exposure and vary across scenarios
    assert data_3["predicted_annual_crashes"] > data_2["predicted_annual_crashes"] > data_1["predicted_annual_crashes"]

    # Verify model version and statistical model metadata are properly populated
    assert data_2["model_version"] in ["2.1.0", "1.0.0"]
    assert "Log-Exposure" in data_2["statistical_model"]

def test_corridor_risk_dynamic_response_to_telematics_and_shockwaves():
    """Verify Corridor Sensor Risk produces dynamic probability shifts under free-flow vs. shockwave conditions."""
    # Scenario A: Free-flow normal traffic (65 mph, low volume, low occupancy)
    res_free = client.post("/api/v1/predictions/risk", json={
        "road": "00I24E",
        "mile": 181.5,
        "speed_mean": 65.0,
        "volume_mean": 8.0,
        "occupancy_mean": 3.5,
        "weather": "Clear",
        "light": "Daylight",
        "hour": 14
    })
    assert res_free.status_code == 200
    data_free = res_free.json()

    # Scenario B: Moderate congestion
    res_mod = client.post("/api/v1/predictions/risk", json={
        "road": "00I24E",
        "mile": 181.5,
        "speed_mean": 42.0,
        "volume_mean": 18.5,
        "occupancy_mean": 12.0,
        "weather": "Rain",
        "light": "Daylight",
        "hour": 17
    })
    assert res_mod.status_code == 200
    data_mod = res_mod.json()

    # Scenario C: Severe deceleration shockwave (high speed drop, heavy occupancy)
    res_shock = client.post("/api/v1/predictions/risk", json={
        "road": "00I24E",
        "mile": 181.5,
        "speed_mean": 18.0,
        "speed_drop": 25.0,
        "volume_mean": 35.0,
        "occupancy_mean": 55.0,
        "weather": "Snow",
        "light": "Dark",
        "hour": 22
    })
    assert res_shock.status_code == 200
    data_shock = res_shock.json()

    # Probability must change across telemetry regimes
    assert data_free["risk_score"] < data_mod["risk_score"] < data_shock["risk_score"]
    assert data_shock["risk_score"] > 0.05

def test_traffic_severity_imperial_unit_normalization_and_dynamic_response():
    """Verify Traffic Impact Severity normalizes imperial weather without pressure explosion and responds dynamically."""
    # Scenario A: Sunny Los Angeles Interstate junction
    res_la = client.post("/api/v1/predictions/severity", json={
        "latitude": 34.0522,
        "longitude": -118.2437,
        "units": "imperial",
        "temperature": 75.0,
        "humidity": 40.0,
        "pressure": 29.95,
        "visibility": 10.0,
        "wind_speed": 5.0,
        "weather_condition": "Clear",
        "is_junction": True,
        "is_traffic_signal": True
    })
    assert res_la.status_code == 200
    data_la = res_la.json()
    probs_la = data_la["severity_probabilities"]

    # Scenario B: Severe Winter Weather in Seattle non-junction
    res_sea = client.post("/api/v1/predictions/severity", json={
        "latitude": 47.6062,
        "longitude": -122.3321,
        "units": "imperial",
        "temperature": 28.0,
        "humidity": 85.0,
        "pressure": 29.80,
        "visibility": 2.0,
        "wind_speed": 22.0,
        "precipitation": 0.35,
        "weather_condition": "Snow",
        "is_junction": False,
        "is_traffic_signal": False
    })
    assert res_sea.status_code == 200
    data_sea = res_sea.json()
    probs_sea = data_sea["severity_probabilities"]

    # Probabilities must differ between distinct geographic and atmospheric regimes
    assert probs_la["class_2"] != probs_sea["class_2"]
    assert data_la["predicted_severity"] in [1, 2, 3, 4]
    assert data_sea["predicted_severity"] in [1, 2, 3, 4]


def test_corridor_risk_canonicalization_and_shockwave_semantics():
    """Verify Corridor Sensor Risk correctly canonicalizes road/weather/light and handles speed drop deceleration."""
    # Test 1: Canonicalization of lowercase / alias inputs
    res_canon = client.post("/api/v1/predictions/risk", json={
        "road": "i24e",
        "mile": 181.5,
        "speed_mean": 45.0,
        "speed_drop": 20.0,  # Positive magnitude representing 20 mph drop
        "volume_mean": 20.0,
        "occupancy_mean": 25.0,
        "weather": "rain",
        "light": "dark",
        "hour": 20
    })
    assert res_canon.status_code == 200
    data_canon = res_canon.json()
    assert data_canon["road"] == "00I24E"
    assert data_canon["risk_score"] > 0.065
    assert data_canon["accident_alert"] is True
    assert data_canon["risk_level"] in ["HIGH", "CRITICAL"]

    # Test 2: Equivalent signed negative speed drop produces identical shockwave prediction
    res_signed = client.post("/api/v1/predictions/risk", json={
        "road": "00I24E",
        "mile": 181.5,
        "speed_mean": 45.0,
        "speed_drop": -20.0,  # Negative signed delta
        "volume_mean": 20.0,
        "occupancy_mean": 25.0,
        "weather": "Rain",
        "light": "Dark",
        "hour": 20
    })
    assert res_signed.status_code == 200
    data_signed = res_signed.json()
    assert data_signed["risk_score"] == data_canon["risk_score"]
    assert data_signed["risk_level"] == data_canon["risk_level"]
    assert data_signed["accident_alert"] == data_canon["accident_alert"]

