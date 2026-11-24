import os
import sys
import time
import json
import joblib
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

import polars as pl
import pandas as pd
import numpy as np
import mlflow
import lightgbm as lgb
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    confusion_matrix,
    brier_score_loss
)

PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "us_accidents_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
SEVERITY_MODEL_PATH = MODELS_DIR / "accident_severity_lgbm.pkl"
SEVERITY_DUMMY_PATH = MODELS_DIR / "severity_dummy_baseline.pkl"
SEVERITY_METADATA_PATH = MODELS_DIR / "severity_model_metadata.json"
SEVERITY_SCHEMA_PATH = MODELS_DIR / "severity_feature_schema.json"

RAW_FEATURE_COLS = [
    "latitude", "longitude", "hour", "temperature_c", "humidity_pct",
    "pressure_hpa", "visibility_km", "wind_speed_kmh", "precipitation_mm",
    "weather_category", "is_night", "is_amenity", "is_crossing",
    "is_give_way", "is_junction", "is_no_exit", "is_railway",
    "is_roundabout", "is_station", "is_stop_sign", "is_traffic_calming",
    "is_traffic_signal"
]

SEVERITY_MODEL_FEATURES = [c for c in RAW_FEATURE_COLS if c != 'hour'] + ['hour_sin', 'hour_cos']

HELD_OUT_GEOGRAPHIC_STATES = ["WA", "CO", "OH", "GA"]

def load_and_prepare_split(filter_expr):
    df_pl = pl.scan_parquet(PARQUET_PATH).filter(filter_expr).collect()
    df_pd = df_pl.to_pandas()
    h = df_pd['hour'].values
    df_pd['hour_sin'] = np.sin(2 * np.pi * h / 24.0).astype(np.float32)
    df_pd['hour_cos'] = np.cos(2 * np.pi * h / 24.0).astype(np.float32)
    X = df_pd[SEVERITY_MODEL_FEATURES]
    y = df_pd['severity_label'].values
    return X, y

def compute_multiclass_metrics(y_true, y_pred, y_proba, labels=[0, 1, 2, 3]):
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, target_names=["Sev 1", "Sev 2", "Sev 3", "Sev 4"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Log loss (handle cases where not all classes appear in y_true, e.g. 2023)
    try:
        ll = float(log_loss(y_true, y_proba, labels=labels))
    except Exception:
        ll = None

    # Brier score across classes (mean squared error of probability vector to one-hot target)
    y_onehot = np.zeros((len(y_true), len(labels)))
    for i, val in enumerate(y_true):
        if val in labels:
            y_onehot[i, val] = 1.0
    brier = float(np.mean(np.sum((y_proba - y_onehot)**2, axis=1)))

    return {
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "log_loss": round(ll, 4) if ll is not None else None,
        "brier_score": round(brier, 4),
        "per_class": {
            "severity_1": {
                "precision": round(report["Sev 1"]["precision"], 4),
                "recall": round(report["Sev 1"]["recall"], 4),
                "f1": round(report["Sev 1"]["f1-score"], 4),
                "support": int(report["Sev 1"]["support"])
            },
            "severity_2": {
                "precision": round(report["Sev 2"]["precision"], 4),
                "recall": round(report["Sev 2"]["recall"], 4),
                "f1": round(report["Sev 2"]["f1-score"], 4),
                "support": int(report["Sev 2"]["support"])
            },
            "severity_3": {
                "precision": round(report["Sev 3"]["precision"], 4),
                "recall": round(report["Sev 3"]["recall"], 4),
                "f1": round(report["Sev 3"]["f1-score"], 4),
                "support": int(report["Sev 3"]["support"])
            },
            "severity_4": {
                "precision": round(report["Sev 4"]["precision"], 4),
                "recall": round(report["Sev 4"]["recall"], 4),
                "f1": round(report["Sev 4"]["f1-score"], 4),
                "support": int(report["Sev 4"]["support"])
            }
        },
        "confusion_matrix": cm
    }

def train_severity_model():
    print("=" * 70)
    print("PHASE 2: GLOBAL ACCIDENT SEVERITY MODEL TRAINING & EVALUATION")
    print("=" * 70)

    t0 = time.time()
    
    # 1. Load splits
    print("\n[1/5] Loading Chronological and Geographic Splits from Parquet...")
    train_filter = (pl.col("year") <= 2021) & (~pl.col("state").is_in(HELD_OUT_GEOGRAPHIC_STATES))
    val_filter = (pl.col("year") == 2022) & (~pl.col("state").is_in(HELD_OUT_GEOGRAPHIC_STATES))
    geo_filter = (pl.col("year") >= 2021) & (pl.col("state").is_in(HELD_OUT_GEOGRAPHIC_STATES))
    test_filter = (pl.col("year") == 2023)

    X_train, y_train = load_and_prepare_split(train_filter)
    X_val, y_val = load_and_prepare_split(val_filter)
    X_geo, y_geo = load_and_prepare_split(geo_filter)
    X_test, y_test = load_and_prepare_split(test_filter)

    print(f"  Training Split (2016-2021):             {len(X_train):,} samples")
    print(f"  Validation Split (2022):                {len(X_val):,} samples")
    print(f"  Geographic Holdout (WA, CO, OH, GA):    {len(X_geo):,} samples")
    print(f"  Chronological Test Split (2023):        {len(X_test):,} samples")

    # 2. Train Dummy Baseline
    print("\n[2/5] Training DummyClassifier Baseline...")
    dummy = DummyClassifier(strategy='prior', random_state=42)
    dummy.fit(X_train, y_train)
    dummy_val_proba = dummy.predict_proba(X_val)
    dummy_val_preds = dummy.predict(X_val)
    dummy_metrics = compute_multiclass_metrics(y_val, dummy_val_preds, dummy_val_proba)
    print(f"  Dummy (Val 2022): Macro F1={dummy_metrics['macro_f1']}, Weighted F1={dummy_metrics['weighted_f1']}, BalAcc={dummy_metrics['balanced_accuracy']}")

    # 3. Train SGD Linear Baseline
    print("\n[3/5] Training SGDClassifier (Linear Multiclass) Baseline...")
    sgd = SGDClassifier(loss='log_loss', class_weight='balanced', max_iter=25, random_state=42)
    sgd.fit(X_train, y_train)
    sgd_val_proba = sgd.predict_proba(X_val)
    sgd_val_preds = sgd.predict(X_val)
    sgd_metrics = compute_multiclass_metrics(y_val, sgd_val_preds, sgd_val_proba)
    print(f"  SGD (Val 2022):   Macro F1={sgd_metrics['macro_f1']}, Weighted F1={sgd_metrics['weighted_f1']}, BalAcc={sgd_metrics['balanced_accuracy']}")

    # 4. Train Optimized LightGBM Multiclass Model
    print("\n[4/5] Training LightGBM Multiclass Model on 5.3M records...")
    t_lgb = time.time()
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 4,
        'class_weight': 'balanced',
        'n_estimators': 120,
        'learning_rate': 0.08,
        'num_leaves': 31,
        'max_depth': 6,
        'subsample': 0.85,
        'colsample_bytree': 0.80,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)
    print(f"  Fitted LightGBM in {time.time() - t_lgb:.1f}s!")

    # 5. Evaluate on all evaluation holdouts
    print("\n[5/5] Comprehensive Multi-Split Evaluation...")
    val_proba = lgb_model.predict_proba(X_val)
    val_preds = np.argmax(val_proba, axis=1)
    val_metrics = compute_multiclass_metrics(y_val, val_preds, val_proba)

    geo_proba = lgb_model.predict_proba(X_geo)
    geo_preds = np.argmax(geo_proba, axis=1)
    geo_metrics = compute_multiclass_metrics(y_geo, geo_preds, geo_proba)

    test_proba = lgb_model.predict_proba(X_test)
    test_preds = np.argmax(test_proba, axis=1)
    test_metrics = compute_multiclass_metrics(y_test, test_preds, test_proba)

    print("\n" + "=" * 70)
    print("SEVERITY MODEL TEST & HOLDOUT PERFORMANCE SUMMARY:")
    print("=" * 70)
    print(f"{'Split':<25} | {'Macro F1':<10} | {'Weighted F1':<12} | {'Bal Acc':<10} | {'Sev4 Recall':<12}")
    print("-" * 75)
    print(f"{'Validation 2022 (All 4)':<25} | {val_metrics['macro_f1']:<10.4f} | {val_metrics['weighted_f1']:<12.4f} | {val_metrics['balanced_accuracy']:<10.4f} | {val_metrics['per_class']['severity_4']['recall']:<12.4f}")
    print(f"{'Geographic (WA,CO,OH,GA)':<25} | {geo_metrics['macro_f1']:<10.4f} | {geo_metrics['weighted_f1']:<12.4f} | {geo_metrics['balanced_accuracy']:<10.4f} | {geo_metrics['per_class']['severity_4']['recall']:<12.4f}")
    print(f"{'Chronological Test 2023':<25} | {test_metrics['macro_f1']:<10.4f} | {test_metrics['weighted_f1']:<12.4f} | {test_metrics['balanced_accuracy']:<10.4f} | {test_metrics['per_class']['severity_4']['recall']:<12.4f}")
    print("=" * 70)

    # Feature Importances
    importances = lgb_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("\nTop 10 Features in Severity Model:")
    for idx in top_indices:
        print(f"  {SEVERITY_MODEL_FEATURES[idx]:25}: {importances[idx]}")

    # Track in MLflow
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("accident-severity-prediction-phase2")
    with mlflow.start_run(run_name="lightgbm_multiclass_severity_v1") as run:
        mlflow.log_params(lgb_params)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("val_samples", len(X_val))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_metric("val_macro_f1", val_metrics["macro_f1"])
        mlflow.log_metric("val_weighted_f1", val_metrics["weighted_f1"])
        mlflow.log_metric("val_balanced_accuracy", val_metrics["balanced_accuracy"])
        mlflow.log_metric("val_sev4_recall", val_metrics["per_class"]["severity_4"]["recall"])
        mlflow.log_metric("geo_macro_f1", geo_metrics["macro_f1"])
        mlflow.log_metric("geo_balanced_accuracy", geo_metrics["balanced_accuracy"])
        mlflow.log_metric("test_macro_f1", test_metrics["macro_f1"])
        mlflow.log_metric("test_weighted_f1", test_metrics["weighted_f1"])
        mlflow.log_metric("test_balanced_accuracy", test_metrics["balanced_accuracy"])
        mlflow.log_metric("test_sev4_recall", test_metrics["per_class"]["severity_4"]["recall"])

    # Save artifacts
    print("\nSaving severity model artifacts...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgb_model, SEVERITY_MODEL_PATH)
    joblib.dump(dummy, SEVERITY_DUMMY_PATH)

    schema_info = {
        "model_type": "multiclass_classification",
        "num_classes": 4,
        "classes": [
            {"id": 1, "code": "Sev 1", "label": "MINIMAL_IMPACT", "description": "Minimal delay / local traffic impact"},
            {"id": 2, "code": "Sev 2", "label": "MODERATE_IMPACT", "description": "Short delay / standard traffic blockage"},
            {"id": 3, "code": "Sev 3", "label": "SIGNIFICANT_IMPACT", "description": "Extended delay / major corridor blockage"},
            {"id": 4, "code": "Sev 4", "label": "SEVERE_DISRUPTION", "description": "Corridor closure / severe roadway disruption"}
        ],
        "feature_columns": SEVERITY_MODEL_FEATURES,
        "feature_count": len(SEVERITY_MODEL_FEATURES),
        "measurement_system": "canonical_metric (Celsius, km/h, km, hPa, mm)"
    }
    with open(SEVERITY_SCHEMA_PATH, "w") as f:
        json.dump(schema_info, f, indent=2)

    metadata = {
        "model_name": "accident_severity_lightgbm",
        "model_version": "2.0.0",
        "trained_at": datetime.utcnow().isoformat(),
        "task": "Multiclass Accident Traffic-Impact Severity Prediction (1-4)",
        "training_data_source": "US Accidents March 2023 Release (Moosavi et al., 7.728M records)",
        "training_geography": "United States (49 contiguous states + DC)",
        "licensing_and_commercial_terms": {
            "dataset_license": "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
            "commercial_status": "Academic Research & Pre-Commercial SaaS Validation. Commercial production deployment requires licensing underlying telematics data (e.g. TomTom, HERE, INRIX) or training on public-domain CC-0 crash datasets.",
            "derivative_restriction": "Models trained directly on CC BY-NC-SA 4.0 data share non-commercial attribution clauses."
        },
        "sample_counts": {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "geographic_holdout_samples": len(X_geo),
            "test_samples": len(X_test),
            "total_samples_processed": len(X_train) + len(X_val) + len(X_geo) + len(X_test)
        },
        "hyperparameters": lgb_params,
        "dummy_baseline_metrics": dummy_metrics,
        "sgd_baseline_metrics": sgd_metrics,
        "validation_metrics": val_metrics,
        "geographic_holdout_metrics": geo_metrics,
        "test_metrics": test_metrics,
        "top_features": [
            {"feature": SEVERITY_MODEL_FEATURES[idx], "importance": int(importances[idx])}
            for idx in top_indices
        ]
    }
    with open(SEVERITY_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts successfully saved to {MODELS_DIR}")
    print(f"Total pipeline execution time: {time.time() - t0:.1f}s")
    return metadata

if __name__ == "__main__":
    train_severity_model()
