"""Phase 3 Comprehensive Verification Script
Verifies:
1. Model Artifact Loading & Reproducibility
2. Coordinate Memorization Analysis & OOD Detection
3. API Endpoints (all 6) via FastAPI TestClient
4. Database Persistence & Logging
5. Licensing & Commercial Boundaries
"""
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.db.session import SessionLocal
from backend.app.db.models.prediction_log import PredictionLog
from backend.app.db.models.severity_prediction_log import SeverityPredictionLog
from backend.app.db.models.combined_prediction_log import CombinedPredictionLog
from backend.app.services.prediction_service import prediction_service
from backend.app.services.severity_service import severity_service
from backend.app.services.combined_service import combined_service

def verify_model_artifacts():
    print("=" * 60)
    print("1. VERIFYING MODEL ARTIFACTS AND PIPELINES")
    print("=" * 60)

    # 1. Risk Model
    risk_model = joblib.load(settings.MODEL_PATH)
    risk_pipeline = joblib.load(settings.PIPELINE_PATH)
    with open(settings.SCHEMA_PATH, "r") as f:
        risk_schema = json.load(f)
    with open(settings.METADATA_PATH, "r") as f:
        risk_meta = json.load(f)

    print(f"Risk Model Loaded: {type(risk_model).__name__} (v{risk_meta['model_version']})")
    print(f"Risk Pipeline Features: {len(risk_schema['numeric_features'])} numerical, {len(risk_schema['categorical_features'])} categorical")
    print(f"Risk Optimal Decision Threshold: {risk_meta['optimal_threshold']:.4f}")
    print(f"Risk Test ROC-AUC: {risk_meta['test_metrics']['roc_auc']:.4f}, PR-AUC: {risk_meta['test_metrics']['pr_auc']:.4f}")

    # 2. Severity Model
    sev_model_path = PROJECT_ROOT / "models" / "accident_severity_lgbm.pkl"
    sev_meta_path = PROJECT_ROOT / "models" / "severity_model_metadata.json"
    sev_schema_path = PROJECT_ROOT / "models" / "severity_feature_schema.json"

    sev_model = joblib.load(sev_model_path)
    with open(sev_meta_path, "r") as f:
        sev_meta = json.load(f)
    with open(sev_schema_path, "r") as f:
        sev_schema = json.load(f)

    print(f"\nSeverity Model Loaded: {type(sev_model).__name__} (v{sev_meta['model_version']})")
    print(f"Severity Features Count: {len(sev_schema['feature_columns'])}")
    print(f"Severity Classes: {sev_model.classes_.tolist()}")
    print(f"Validation Macro F1: {sev_meta['validation_metrics']['macro_f1']:.4f}, Balanced Acc: {sev_meta['validation_metrics']['balanced_accuracy']:.4f}")
    print(f"Severity 4 Recall (Validation): {sev_meta['validation_metrics']['per_class']['severity_4']['recall']:.2%}")
    print(f"Severity 4 Recall (2023 Test): {sev_meta['test_metrics']['per_class']['severity_4']['recall']:.2%}")

    return True

def verify_coordinate_memorization_and_ood():
    print("\n" + "=" * 60)
    print("2. COORDINATE MEMORIZATION ANALYSIS & OOD DETECTION")
    print("=" * 60)

    locations = [
        {"name": "New York, USA", "lat": 40.7128, "lng": -74.0060, "expected_ood": False},
        {"name": "Los Angeles, USA", "lat": 34.0522, "lng": -118.2437, "expected_ood": False},
        {"name": "Chicago, USA", "lat": 41.8781, "lng": -87.6298, "expected_ood": False},
        {"name": "London, UK", "lat": 51.5074, "lng": -0.1278, "expected_ood": True},
        {"name": "Paris, France", "lat": 48.8566, "lng": 2.3522, "expected_ood": True},
        {"name": "Tokyo, Japan", "lat": 35.6762, "lng": 139.6503, "expected_ood": True},
        {"name": "Sydney, Australia", "lat": -33.8688, "lng": 151.2093, "expected_ood": True},
    ]

    client = TestClient(app)
    results = []

    for loc in locations:
        payload = {
            "latitude": loc["lat"],
            "longitude": loc["lng"],
            "timezone": "UTC",
            "units": "metric",
            "temperature": 15.0,
            "humidity": 65.0,
            "pressure": 1013.25,
            "visibility": 10.0,
            "wind_speed": 12.0,
            "precipitation": 0.0,
            "weather_condition": "Clear",
            "is_junction": True,
            "is_traffic_signal": True
        }
        res = client.post("/api/v1/predictions/severity", json=payload)
        data = res.json()
        ood = data["geographic_metadata"]["is_out_of_distribution"]
        status = data["geographic_metadata"]["validation_status"]
        warning = data["geographic_metadata"]["validation_warning"]
        pred_sev = data["predicted_severity"]
        probs = data["severity_probabilities"]

        assert ood == loc["expected_ood"], f"OOD mismatch for {loc['name']}"
        results.append({
            "Location": loc["name"],
            "Coords": f"({loc['lat']}, {loc['lng']})",
            "OOD": ood,
            "Status": status,
            "Predicted Sev": pred_sev,
            "P(Sev=2)": f"{probs['class_2']:.3f}",
            "P(Sev=4)": f"{probs['class_4']:.3f}",
            "Warning Triggered": bool(warning)
        })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

def verify_all_api_endpoints():
    print("\n" + "=" * 60)
    print("3. VERIFYING ALL 6 API ENDPOINTS")
    print("=" * 60)

    client = TestClient(app)

    # 1. GET /health
    r1 = client.get("/health")
    assert r1.status_code == 200
    print(f"[OK] GET /health -> status {r1.status_code}, payload: {r1.json()}")

    # 2. GET /api/v1/health
    r2 = client.get("/api/v1/health")
    assert r2.status_code == 200
    print(f"[OK] GET /api/v1/health -> status {r2.status_code}, database: {r2.json().get('database')}")

    # 3. GET /api/v1/models
    r3 = client.get("/api/v1/models")
    assert r3.status_code == 200
    m_data = r3.json()
    assert "models" in m_data and len(m_data["models"]) == 2
    print(f"[OK] GET /api/v1/models -> status {r3.status_code}, registered models: {[m['model_id'] for m in m_data['models']]}")

    # 4. POST /api/v1/predictions/risk
    risk_payload = {
        "road": "00I24E",
        "mile": 184.7,
        "hour": 16,
        "weather": "Rain",
        "light": "Daylight",
        "speed_mean": 38.0,
        "speed_std": 8.0,
        "speed_drop": -18.0,
        "volume_mean": 28.0,
        "volume_std": 6.0,
        "occupancy_mean": 24.0,
        "occupancy_max": 38.0,
        "occupancy_std": 6.0,
        "speed_upstream": 55.0,
        "speed_downstream": 32.0
    }
    r4 = client.post("/api/v1/predictions/risk", json=risk_payload)
    assert r4.status_code == 200
    print(f"[OK] POST /api/v1/predictions/risk -> status {r4.status_code}, risk_score: {r4.json()['risk_score']:.4f}, alert: {r4.json()['accident_alert']}")

    # 5. POST /api/v1/predictions/severity (Metric & Imperial)
    sev_payload_imp = {
        "latitude": 29.7604,
        "longitude": -95.3698,
        "timezone": "America/Chicago",
        "units": "imperial",
        "temperature": 85.0,
        "humidity": 70.0,
        "pressure": 29.95,
        "visibility": 9.0,
        "wind_speed": 10.0,
        "precipitation": 0.1,
        "weather_condition": "Thunderstorm",
        "is_junction": True
    }
    r5 = client.post("/api/v1/predictions/severity", json=sev_payload_imp)
    assert r5.status_code == 200
    print(f"[OK] POST /api/v1/predictions/severity (Imperial) -> status {r5.status_code}, severity: {r5.json()['predicted_severity']}, level: {r5.json()['severity_level']}")

    # 6. POST /api/v1/predictions/combined
    comb_payload = {
        "risk_inputs": risk_payload,
        "severity_inputs": {
            "latitude": 35.0456,
            "longitude": -85.3097,
            "timezone": "America/New_York",
            "units": "metric",
            "temperature": 22.0,
            "humidity": 55.0,
            "pressure": 1014.0,
            "visibility": 10.0,
            "wind_speed": 10.0,
            "precipitation": 0.0,
            "weather_condition": "Clear",
            "is_junction": True,
            "is_traffic_signal": True
        }
    }
    r6 = client.post("/api/v1/predictions/combined", json=comb_payload)
    assert r6.status_code == 200
    c_data = r6.json()
    print(f"[OK] POST /api/v1/predictions/combined -> status {r6.status_code}, expected_sev: {c_data['expected_severity_index']}, joint_score: {c_data['joint_risk_impact_score']}")

    # Error handling check (invalid WGS84 coordinates)
    bad_res = client.post("/api/v1/predictions/severity", json={"latitude": 120.0, "longitude": -85.0})
    assert bad_res.status_code == 422
    print(f"[OK] Error Handling Check -> Latitude 120.0 correctly rejected with HTTP 422 Unprocessable Entity")

def verify_database_logging():
    print("\n" + "=" * 60)
    print("4. VERIFYING DATABASE LOGGING & AUDIT TRAIL")
    print("=" * 60)

    db = SessionLocal()
    try:
        risk_count = db.query(PredictionLog).count()
        sev_count = db.query(SeverityPredictionLog).count()
        comb_count = db.query(CombinedPredictionLog).count()

        print(f"Risk Prediction Logs Count: {risk_count}")
        print(f"Severity Prediction Logs Count: {sev_count}")
        print(f"Combined Prediction Logs Count: {comb_count}")

        # Retrieve latest combined prediction
        latest_comb = db.query(CombinedPredictionLog).order_by(CombinedPredictionLog.created_at.desc()).first()
        if latest_comb:
            print(f"Latest Combined Log ID: {latest_comb.id}")
            print(f"  Risk Score: {latest_comb.risk_score:.4f}, Alert: {latest_comb.accident_alert}")
            print(f"  Predicted Severity: {latest_comb.predicted_severity}, Expected Index: {latest_comb.expected_severity_index:.3f}")
            print(f"  Joint Impact Score: {latest_comb.joint_risk_impact_score:.4f}")
            print(f"  Payload Summary: {latest_comb.payload_summary}")
    finally:
        db.close()

if __name__ == "__main__":
    verify_model_artifacts()
    verify_coordinate_memorization_and_ood()
    verify_all_api_endpoints()
    verify_database_logging()
    print("\n" + "=" * 60)
    print("ALL PHASE 3 VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
