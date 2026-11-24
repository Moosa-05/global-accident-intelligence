"""Phase 7: End-to-End Verification of All Five Prediction Routes & Timescale Separation.
Ensures:
1. Micro-scale corridor sensor risk (/predictions/risk)
2. Point-level traffic delay severity (/predictions/severity)
3. Combined risk and delay severity (/predictions/combined)
4. Human physiological trauma severity (/predictions/injury-severity)
5. Macro-scale road segment annual exposure risk (/predictions/exposure-risk)
"""
from fastapi.testclient import TestClient

def test_all_five_prediction_routes_and_timescales(client: TestClient):
    # 1. Micro-scale immediate sensor risk (15-min probability)
    r1 = client.post("/api/v1/predictions/risk", json={
        "road": "00I24E",
        "mile": 184.7,
        "hour": 14,
        "weather": "Clear",
        "light": "Daylight",
        "speed_mean": 42.0,
        "speed_std": 6.5,
        "speed_drop": -14.2,
        "volume_mean": 22.0,
        "volume_std": 5.1,
        "occupancy_mean": 18.0,
        "occupancy_max": 28.5,
        "occupancy_std": 4.8,
        "speed_upstream": 54.0,
        "speed_downstream": 38.0
    })
    assert r1.status_code == 200
    res1 = r1.json()
    assert 0.0 <= res1["risk_score"] <= 1.0
    assert "risk_level" in res1
    assert "accident_alert" in res1

    # 2. Traffic delay severity (Classes 1-4)
    r2 = client.post("/api/v1/predictions/severity", json={
        "latitude": 35.0456,
        "longitude": -85.3097,
        "timezone": "America/New_York",
        "units": "metric",
        "temperature": 22.0,
        "is_junction": True
    })
    assert r2.status_code == 200
    res2 = r2.json()
    assert res2["predicted_severity"] in [1, 2, 3, 4]
    assert "severity_probabilities" in res2
    assert "geographic_metadata" in res2

    # 3. Combined risk + severity
    r3 = client.post("/api/v1/predictions/combined", json={
        "risk_inputs": {
            "road": "00I24E",
            "mile": 184.7,
            "hour": 14,
            "weather": "Clear",
            "light": "Daylight",
            "speed_mean": 42.0,
            "speed_std": 6.5,
            "speed_drop": -14.2,
            "volume_mean": 22.0,
            "volume_std": 5.1,
            "occupancy_mean": 18.0,
            "occupancy_max": 28.5,
            "occupancy_std": 4.8,
            "speed_upstream": 54.0,
            "speed_downstream": 38.0
        },
        "severity_inputs": {
            "latitude": 35.0456,
            "longitude": -85.3097,
            "timezone": "America/New_York",
            "units": "metric",
            "temperature": 22.0,
            "is_junction": True
        }
    })
    assert r3.status_code == 200
    res3 = r3.json()
    assert "accident_risk" in res3
    assert "conditional_severity" in res3
    assert "joint_risk_impact_score" in res3

    # 4. Human Injury Severity (3-Class: SLIGHT, SERIOUS, FATAL)
    r4 = client.post("/api/v1/predictions/injury-severity", json={
        "latitude": 51.5074,
        "longitude": -0.1278,
        "timezone": "Europe/London",
        "units": "metric",
        "road_class": "PRIMARY_ARTERIAL",
        "speed_limit_kmh": 48.0,
        "temperature": 15.0,
        "weather_condition": "CLEAR"
    })
    assert r4.status_code == 200
    res4 = r4.json()
    assert res4["predicted_severity"] in ["SLIGHT", "SERIOUS", "FATAL"]
    assert "probabilities" in res4
    assert "fatality_risk_alert" in res4
    assert "Human Physiological Injury Severity" in res4["target_definition"]

    # 5. Macro-scale road segment annual exposure risk
    r5 = client.post("/api/v1/predictions/exposure-risk", json={
        "road_name": "M25",
        "road_class": "MOTORWAY",
        "link_length_km": 8.0,
        "speed_limit_kmh": 112.7,
        "latitude": 51.5,
        "longitude": 0.0,
        "annual_average_daily_flow": 120000
    })
    assert r5.status_code == 200
    res5 = r5.json()
    assert res5["predicted_annual_crashes"] >= 0.0
    assert res5["predicted_crash_rate_per_million_vkt"] >= 0.0
    assert res5["risk_tier"] in ["LOW_RISK", "TYPICAL_RISK", "ELEVATED_RISK", "HIGH_RISK"]
    assert "Macro-scale annual road segment model" in res5["exposure_interpretation_notice"]

    # Assert strict timescale separation
    # Micro risk is an instantaneous 15-minute event probability [0, 1]
    assert 0.0 <= res1["risk_score"] <= 1.0
    # Macro risk is an annual count and rate per MVKT
    assert res5["predicted_annual_crashes"] > 0.0
