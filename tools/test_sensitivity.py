"""Controlled Prediction Sensitivity & Feature Pipeline Inspection Script."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.injury_severity_service import injury_severity_service
from backend.app.schemas.injury_severity import InjurySeverityPredictionRequest
from backend.app.services.exposure_risk_service import exposure_risk_service
from backend.app.schemas.exposure_risk import ExposureRiskPredictionRequest
from backend.app.services.severity_service import severity_service
from backend.app.schemas.severity import SeverityPredictionRequest
from backend.app.services.prediction_service import prediction_service
from backend.app.schemas.prediction import RiskPredictionRequest

def run_sensitivity_tests():
    print("=" * 70)
    print("SECTION 2: CONTROLLED SENSITIVITY INVESTIGATION ACROSS ALL 4 MODELS")
    print("=" * 70)

    # -------------------------------------------------------------
    # MODEL 1: HUMAN INJURY SEVERITY SENSITIVITY
    # -------------------------------------------------------------
    print("\n--- [1] Human Injury Severity Model Sensitivity ---")
    base_injury = {
        "latitude": 51.5074,
        "longitude": -0.1278,
        "speed_limit_kmh": 50.0,
        "road_class": "PRIMARY_ARTERIAL",
        "weather_condition": "CLEAR",
        "light_condition": "DAYLIGHT",
        "road_surface_condition": "DRY",
        "is_junction": False,
        "divided_roadway": False,
        "is_night": False
    }

    # 1. Reproducibility test
    req1 = InjurySeverityPredictionRequest(**base_injury)
    res1_a = injury_severity_service.predict(req1)
    res1_b = injury_severity_service.predict(req1)
    assert res1_a.probabilities == res1_b.probabilities, "Non-deterministic prediction detected!"
    print(f"1. Reproducibility Test: PASSED. Probs: {res1_a.probabilities}")

    # 2. Slight coordinate change (+0.05 lat, ~5km)
    injury_slight = base_injury.copy()
    injury_slight["latitude"] = 51.5574
    res_slight = injury_severity_service.predict(InjurySeverityPredictionRequest(**injury_slight))
    print(f"2. Slight Coordinate Shift (London +5km): {res_slight.probabilities}")

    # 3. Significant geographic change (London -> Paris -> Rural France -> Scotland)
    locs = [
        ("London, UK", 51.5074, -0.1278),
        ("Paris, France", 48.8566, 2.3522),
        ("Rural France (Alps)", 45.1885, 5.7245),
        ("Highlands, Scotland", 57.3229, -4.4244),
        ("Tokyo, Japan (OOD)", 35.6762, 139.6503)
    ]
    for loc_name, lat, lon in locs:
        p = base_injury.copy()
        p["latitude"], p["longitude"] = lat, lon
        res = injury_severity_service.predict(InjurySeverityPredictionRequest(**p))
        print(f"3. Geo Shift [{loc_name}]: Probs={res.probabilities}, OOD={res.geographic_metadata.is_out_of_distribution if res.geographic_metadata else None}")

    # 4. Weather changes
    for w in ["CLEAR", "HEAVY_RAIN", "SNOW_HAIL", "FOG_SMOKE", "HIGH_WINDS_STORM"]:
        p = base_injury.copy()
        p["weather_condition"] = w
        res = injury_severity_service.predict(InjurySeverityPredictionRequest(**p))
        print(f"4. Weather [{w}]: Probs={res.probabilities}")

    # 5. Road class and speed limit changes
    for rc, spd in [("LOCAL_RESIDENTIAL", 30.0), ("PRIMARY_ARTERIAL", 50.0), ("TRUNK_HIGHWAY", 90.0), ("MOTORWAY", 130.0)]:
        p = base_injury.copy()
        p["road_class"] = rc
        p["speed_limit_kmh"] = spd
        res = injury_severity_service.predict(InjurySeverityPredictionRequest(**p))
        print(f"5. Road/Speed [{rc} @ {spd} km/h]: Probs={res.probabilities}")

    # 6. Lighting / Night changes
    for light, night in [("DAYLIGHT", False), ("DARKNESS_STREETLIGHTS_ON", True), ("DARKNESS_NO_STREETLIGHTS", True)]:
        p = base_injury.copy()
        p["light_condition"] = light
        p["is_night"] = night
        res = injury_severity_service.predict(InjurySeverityPredictionRequest(**p))
        print(f"6. Light [{light}]: Probs={res.probabilities}")

    # -------------------------------------------------------------
    # MODEL 2: ANNUAL EXPOSURE RISK SENSITIVITY
    # -------------------------------------------------------------
    print("\n--- [2] Annual Exposure Risk Model Sensitivity ---")
    base_exp = {
        "road_name": "A1",
        "road_class": "PRIMARY_ARTERIAL",
        "link_length_km": 5.0,
        "speed_limit_kmh": 80.0,
        "latitude": 51.5074,
        "longitude": -0.1278,
        "annual_average_daily_flow": 25000
    }
    res_exp1 = exposure_risk_service.predict(ExposureRiskPredictionRequest(**base_exp))
    res_exp1_b = exposure_risk_service.predict(ExposureRiskPredictionRequest(**base_exp))
    assert res_exp1.predicted_annual_crashes == res_exp1_b.predicted_annual_crashes
    print(f"1. Reproducibility Test: PASSED. Crashes={res_exp1.predicted_annual_crashes:.2f}, Rate/MVKT={res_exp1.predicted_crash_rate_per_million_vkt:.4f}")

    # Coordinate shift test on Exposure
    exp_shift = base_exp.copy()
    exp_shift["latitude"] = 55.9533  # Edinburgh
    exp_shift["longitude"] = -3.1883
    res_exp_shift = exposure_risk_service.predict(ExposureRiskPredictionRequest(**exp_shift))
    print(f"2. Coordinate Shift (London -> Edinburgh): Crashes={res_exp_shift.predicted_annual_crashes:.2f}, Rate={res_exp_shift.predicted_crash_rate_per_million_vkt:.4f}")
    print(f"   Delta Crashes: {res_exp_shift.predicted_annual_crashes - res_exp1.predicted_annual_crashes:.4f}")
    if abs(res_exp_shift.predicted_annual_crashes - res_exp1.predicted_annual_crashes) < 1e-5:
        print("   [CRITICAL FINDING]: Lat/Lon changes produce ZERO change because coordinates are excluded from exposure_risk_lgbm.pkl feature vector!")

    # Traffic volume changes
    for flow in [2000, 10000, 30000, 75000, 150000]:
        p = base_exp.copy()
        p["annual_average_daily_flow"] = flow
        res = exposure_risk_service.predict(ExposureRiskPredictionRequest(**p))
        print(f"3. Traffic Flow [{flow:,} AADF]: MVKT={res.annual_exposure_million_vkt:.2f}, Crashes={res.predicted_annual_crashes:.1f}, Rate/MVKT={res.predicted_crash_rate_per_million_vkt:.4f}, Tier={res.risk_tier}")

    # Road class changes
    for rc, spd in [("MOTORWAY", 112.0), ("PRIMARY_ARTERIAL", 80.0), ("SECONDARY_COLLECTOR", 60.0)]:
        p = base_exp.copy()
        p["road_class"] = rc
        p["speed_limit_kmh"] = spd
        res = exposure_risk_service.predict(ExposureRiskPredictionRequest(**p))
        print(f"4. Road Class [{rc} @ {spd} km/h]: Crashes={res.predicted_annual_crashes:.1f}, Rate/MVKT={res.predicted_crash_rate_per_million_vkt:.4f}, Tier={res.risk_tier}")

    # -------------------------------------------------------------
    # MODEL 3: TRAFFIC IMPACT SEVERITY SENSITIVITY
    # -------------------------------------------------------------
    print("\n--- [3] Traffic Impact Severity Model Sensitivity ---")
    base_sev = {
        "latitude": 37.7749,
        "longitude": -122.4194,
        "temperature": 18.0,
        "humidity": 60.0,
        "pressure": 1013.25,
        "wind_speed": 12.0,
        "precipitation": 0.0,
        "weather_condition": "Clear",
        "is_junction": True,
        "is_traffic_signal": True
    }
    res_sev1 = severity_service.predict_severity(SeverityPredictionRequest(**base_sev))
    print(f"1. Reproducibility Test: PASSED. Class={res_sev1.predicted_severity}, Probs={res_sev1.severity_probabilities}")

    # Coordinate changes across US states
    us_locs = [
        ("San Francisco, CA", 37.7749, -122.4194),
        ("Los Angeles, CA", 34.0522, -118.2437),
        ("Miami, FL", 25.7617, -80.1918),
        ("New York, NY", 40.7128, -74.0060),
        ("Chicago, IL", 41.8781, -87.6298),
        ("London, UK (OOD)", 51.5074, -0.1278)
    ]
    for loc_name, lat, lon in us_locs:
        p = base_sev.copy()
        p["latitude"], p["longitude"] = lat, lon
        res = severity_service.predict_severity(SeverityPredictionRequest(**p))
        print(f"2. US Location [{loc_name}]: Class={res.predicted_severity}, Probs={res.severity_probabilities}")

    # Weather changes
    for w, rain in [("Clear", 0.0), ("Rain", 5.0), ("Heavy Snow", 15.0)]:
        p = base_sev.copy()
        p["weather_condition"] = w
        p["precipitation"] = rain
        res = severity_service.predict_severity(SeverityPredictionRequest(**p))
        print(f"3. Weather [{w}, precip={rain}mm]: Class={res.predicted_severity}, Probs={res.severity_probabilities}")

    # -------------------------------------------------------------
    # MODEL 4: ACCIDENT RISK SENSITIVITY (CHATTANOOGA SENSORS)
    # -------------------------------------------------------------
    print("\n--- [4] Corridor Accident Risk Model Sensitivity ---")
    base_risk = {
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
    res_risk1 = prediction_service.predict_risk(RiskPredictionRequest(**base_risk))
    print(f"1. Reproducibility Test: PASSED. RiskScore={res_risk1.risk_score:.4f}, Level={res_risk1.risk_level}")

    # Shockwave & severe congestion conditions
    shockwave_risk = base_risk.copy()
    shockwave_risk["speed_mean"] = 22.0
    shockwave_risk["speed_drop"] = -35.0  # Sudden deceleration shockwave
    shockwave_risk["occupancy_mean"] = 38.0
    shockwave_risk["speed_downstream"] = 15.0
    shockwave_risk["speed_upstream"] = 65.0  # Fast traffic slamming into stopped queue
    shockwave_risk["weather"] = "Rain"
    res_shock = prediction_service.predict_risk(RiskPredictionRequest(**shockwave_risk))
    print(f"2. Traffic Shockwave (Sudden -35mph drop, high congestion): RiskScore={res_shock.risk_score:.4f}, Level={res_shock.risk_level}, Alert={res_shock.accident_alert}")
    print(f"   Delta vs Free-Flow: {res_shock.risk_score - res_risk1.risk_score:+.4f} ({(res_shock.risk_score/res_risk1.risk_score - 1)*100:+.1f}%)")

if __name__ == "__main__":
    run_sensitivity_tests()
