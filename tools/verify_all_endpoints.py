"""Verify all four live prediction endpoints and health routes on the FastAPI backend without authentication.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_endpoints():
    print("=" * 80)
    print("VERIFYING PUBLIC DEMONSTRATION API ENDPOINTS")
    print("=" * 80)
    
    # 1. Health checks
    r = client.get("/health/live")
    print(f"1. /health/live: status={r.status_code}, data={r.json()}")
    assert r.status_code == 200
    
    r = client.get("/health/ready")
    print(f"2. /health/ready: status={r.status_code}, data={r.json()}")
    assert r.status_code == 200
    
    # 2. Models registry
    r = client.get("/api/v1/models")
    print(f"3. /api/v1/models: status={r.status_code}, model_count={len(r.json()['models'])}")
    assert r.status_code == 200
    
    # 3. Model 1: Human Injury Severity (UK coordinates)
    payload_injury = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "speed_limit_kmh": 50.0,
        "is_junction": True,
        "road_class": "PRIMARY",
        "weather_condition": "CLEAR",
        "light_condition": "DAYLIGHT",
        "road_surface_condition": "DRY"
    }
    r = client.post("/api/v1/predictions/injury-severity", json=payload_injury)
    print(f"4. Human Injury Severity: status={r.status_code}, predicted_severity={r.json().get('predicted_severity')}, probabilities={r.json().get('probabilities')}")
    assert r.status_code == 200
    
    # 4. Model 2: Annual Segment Exposure Risk
    payload_exposure = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "road_class": "MOTORWAY",
        "speed_limit_kmh": 110.0,
        "link_length_km": 2.5,
        "annual_average_daily_flow": 45000
    }
    r = client.post("/api/v1/predictions/exposure-risk", json=payload_exposure)
    print(f"5. Annual Exposure Risk: status={r.status_code}, predicted_annual_crashes={r.json().get('predicted_annual_crashes')}, risk_tier={r.json().get('risk_tier')}")
    assert r.status_code == 200
    
    # 5. Model 3: Traffic Impact Severity (Research Model - US Accidents)
    payload_traffic = {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "temperature_c": 22.0,
        "humidity_pct": 50.0,
        "pressure_hpa": 1013.25,
        "wind_speed_kmh": 10.0,
        "precipitation_mm": 0.0,
        "distance_km": 0.5,
        "is_junction": False,
        "is_traffic_signal": True,
        "is_crossing": False,
        "is_roundabout": False,
        "is_stop_sign": False,
        "is_give_way": False,
        "is_railway": False,
        "is_weekend": False,
        "is_night": False,
        "road_class": "PRIMARY",
        "weather_condition": "CLEAR",
        "light_condition": "DAYLIGHT",
        "road_surface_condition": "DRY"
    }
    r = client.post("/api/v1/predictions/severity", json=payload_traffic)
    print(f"6. Traffic Impact Severity: status={r.status_code}, predicted_severity={r.json().get('predicted_severity')}, probabilities={r.json().get('probabilities')}")
    assert r.status_code == 200

    # 6. Model 4: Short-term Corridor Risk (Sensor Telematics)
    payload_corridor = {
        "road": "00I24E",
        "mile": 184.7,
        "speed_mean": 58.5,
        "speed_std": 3.2,
        "volume_mean": 15.0,
        "volume_std": 4.1,
        "occupancy_mean": 5.2,
        "occupancy_std": 1.8,
        "speed_drop": 5.0,
        "weather": "Clear",
        "light": "Daylight"
    }
    r = client.post("/api/v1/predictions/risk", json=payload_corridor)
    print(f"7. Short-term Corridor Risk: status={r.status_code}, accident_risk_score={r.json().get('accident_risk_score')}, risk_level={r.json().get('risk_level')}")
    assert r.status_code == 200
    
    # 7. Prediction History Anonymous Session Isolation
    r = client.get("/api/v1/history/predictions")
    print(f"8. Anonymous Prediction History: status={r.status_code}, history_count={len(r.json().get('items', []))}")
    assert r.status_code == 200

    print("\nALL 8 API ENDPOINTS VERIFIED OPERATIONAL WITHOUT AUTHENTICATION!")

if __name__ == "__main__":
    test_endpoints()
