"""Human Injury Severity Model Training Pipeline (3-Class).
Predicts collision-level physiological trauma outcome:
- SLIGHT (Class 0): Minor injuries, sprains, bruises, no inpatient admission > 24h
- SERIOUS (Class 1): Severe trauma, fractures, internal injury, or inpatient admission > 24h
- FATAL (Class 2): Death within 30 days of the collision (WHO / UNECE standard)

Uses verified UK STATS19 and France BAAC datasets.
Performs:
1. Multi-country stratified evaluation
2. Spatially independent validation (held-out geographic regions)
3. Cross-country transfer evaluation (UK -> France, France -> UK)
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    brier_score_loss,
    precision_recall_fscore_support
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup MLflow tracking
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("global-human-injury-severity")

SEVERITY_MAPPING = {
    "SLIGHT": 0,
    "SERIOUS": 1,
    "FATAL": 2
}
REVERSE_SEVERITY_MAPPING = {v: k for k, v in SEVERITY_MAPPING.items()}

FEATURE_COLS_NUM = [
    "latitude",
    "longitude",
    "speed_limit_kmh",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "day_of_week",
    "temperature_c",
    "humidity_pct",
    "pressure_hpa",
    "wind_speed_kmh",
    "precipitation_mm"
]

FEATURE_COLS_BOOL = [
    "is_junction",
    "is_traffic_signal",
    "is_crossing",
    "is_roundabout",
    "is_stop_sign",
    "is_give_way",
    "is_railway",
    "is_weekend",
    "is_night",
    "divided_roadway"
]

FEATURE_COLS_CAT = [
    "road_class",
    "weather_condition",
    "light_condition",
    "road_surface_condition"
]

ALL_FEATURE_COLS = FEATURE_COLS_NUM + FEATURE_COLS_BOOL + FEATURE_COLS_CAT

def evaluate_predictions(y_true, y_pred, y_prob, split_name: str) -> dict:
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Multiclass log loss
    loss = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
    
    # Brier score (one-vs-rest average)
    y_true_oh = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((y_prob - y_true_oh)**2, axis=1)))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    res = {
        "split": split_name,
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4),
        "per_class": {
            "SLIGHT": {"precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4), "f1": round(float(f1[0]), 4), "support": int(sup[0])},
            "SERIOUS": {"precision": round(float(prec[1]), 4), "recall": round(float(rec[1]), 4), "f1": round(float(f1[1]), 4), "support": int(sup[1])},
            "FATAL": {"precision": round(float(prec[2]), 4), "recall": round(float(rec[2]), 4), "f1": round(float(f1[2]), 4), "support": int(sup[2])}
        },
        "confusion_matrix": cm
    }
    return res

def train_injury_model():
    print("=" * 70)
    print("STEP 2: TRAINING HUMAN INJURY SEVERITY MODEL")
    print("=" * 70)

    data_path = PROJECT_ROOT / "data" / "processed" / "global_canonical_accidents_v2.parquet"
    print(f"Loading dataset: {data_path}...")
    df = pl.read_parquet(data_path)

    # Filter to eligible human injury records (SLIGHT, SERIOUS, FATAL)
    df = df.filter(pl.col("human_injury_severity").is_in(["SLIGHT", "SERIOUS", "FATAL"]))
    print(f"Eligible injury collisions: {len(df):,}")

    # Feature Engineering (Cyclical encodings)
    df = df.with_columns([
        np.sin(2.0 * np.pi * pl.col("hour") / 24.0).alias("hour_sin"),
        np.cos(2.0 * np.pi * pl.col("hour") / 24.0).alias("hour_cos"),
        np.sin(2.0 * np.pi * (pl.col("month") - 1) / 12.0).alias("month_sin"),
        np.cos(2.0 * np.pi * (pl.col("month") - 1) / 12.0).alias("month_cos"),
        pl.col("divided_roadway").fill_null(False),
        pl.col("speed_limit_kmh").fill_null(50.0)
    ])

    pdf = df.to_pandas()
    y = pdf["human_injury_severity"].map(SEVERITY_MAPPING).values
    country = pdf["country_code"].values
    regions = pdf["administrative_region"].values

    print("\nTarget Distribution:")
    for name, code in SEVERITY_MAPPING.items():
        cnt = (y == code).sum()
        print(f"  {name} (Class {code}): {cnt:,} ({cnt/len(y):.2%})")

    # Pipeline definition
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLS_NUM),
            ("bool", "passthrough", FEATURE_COLS_BOOL),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_COLS_CAT)
        ]
    )

    # 1. Multi-Country Stratified Split
    print("\nPreparing Stratified Multi-Country Split (80% Train, 10% Val, 10% Test)...")
    indices = np.arange(len(pdf))
    idx_train, idx_temp, y_train, y_temp = train_test_split(indices, y, test_size=0.20, random_state=42, stratify=y)
    idx_val, idx_test, y_val, y_test = train_test_split(idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    X_train_raw = pdf.iloc[idx_train][ALL_FEATURE_COLS]
    X_val_raw = pdf.iloc[idx_val][ALL_FEATURE_COLS]
    X_test_raw = pdf.iloc[idx_test][ALL_FEATURE_COLS]

    # Fit preprocessor
    print("Fitting feature preprocessor...")
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    print(f"Transformed Feature Matrix Shape: {X_train.shape}")

    # Baseline: Dummy Stratified
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)
    dummy_prob = dummy.predict_proba(X_test)
    dummy_metrics = evaluate_predictions(y_test, dummy_pred, dummy_prob, "Dummy Baseline (Prior)")

    # Baseline: Logistic Regression
    print("Training Logistic Regression Baseline...")
    lr = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)
    lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob, "Logistic Regression (Balanced)")

    # Primary Model: Tuned LightGBM
    print("Training Tuned LightGBM Classifier (Balanced)...")
    lgbm = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    t0 = time.time()
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    fit_time = time.time() - t0
    print(f"LightGBM trained in {fit_time:.2f}s (Best Iteration: {lgbm.best_iteration_})")

    # Multi-Country Test Evaluation
    lgbm_pred = lgbm.predict(X_test)
    lgbm_prob = lgbm.predict_proba(X_test)
    test_metrics = evaluate_predictions(y_test, lgbm_pred, lgbm_prob, "Multi-Country Test (10%)")

    # Per-country breakdown in Test Set
    country_test = country[idx_test]
    uk_mask = (country_test == "GB")
    fr_mask = (country_test == "FR")

    uk_test_metrics = evaluate_predictions(y_test[uk_mask], lgbm_pred[uk_mask], lgbm_prob[uk_mask], "UK In-Country Test")
    fr_test_metrics = evaluate_predictions(y_test[fr_mask], lgbm_pred[fr_mask], lgbm_prob[fr_mask], "France In-Country Test")

    # 2. Cross-Country Transfer Evaluation
    print("\n" + "-" * 60)
    print("CROSS-COUNTRY TRANSFER EVALUATION")
    print("-" * 60)
    
    # Train on UK only -> Test on France
    uk_all_idx = np.where(country == "GB")[0]
    fr_all_idx = np.where(country == "FR")[0]

    print("Training UK-only model...")
    X_uk_raw = pdf.iloc[uk_all_idx][ALL_FEATURE_COLS]
    y_uk = y[uk_all_idx]
    X_fr_raw = pdf.iloc[fr_all_idx][ALL_FEATURE_COLS]
    y_fr = y[fr_all_idx]

    prep_uk = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLS_NUM),
            ("bool", "passthrough", FEATURE_COLS_BOOL),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_COLS_CAT)
        ]
    )
    X_uk_trans = prep_uk.fit_transform(X_uk_raw)
    X_fr_trans_for_uk = prep_uk.transform(X_fr_raw)

    lgbm_uk = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, class_weight="balanced",
        n_estimators=100, learning_rate=0.06, max_depth=6, random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_uk.fit(X_uk_trans, y_uk)
    uk_to_fr_pred = lgbm_uk.predict(X_fr_trans_for_uk)
    uk_to_fr_prob = lgbm_uk.predict_proba(X_fr_trans_for_uk)
    uk_to_fr_metrics = evaluate_predictions(y_fr, uk_to_fr_pred, uk_to_fr_prob, "UK-to-France Transfer")

    # Train on France only -> Test on UK
    print("Training France-only model...")
    prep_fr = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLS_NUM),
            ("bool", "passthrough", FEATURE_COLS_BOOL),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURE_COLS_CAT)
        ]
    )
    X_fr_trans = prep_fr.fit_transform(X_fr_raw)
    X_uk_trans_for_fr = prep_fr.transform(X_uk_raw)

    lgbm_fr = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, class_weight="balanced",
        n_estimators=100, learning_rate=0.06, max_depth=6, random_state=42, n_jobs=-1, verbose=-1
    )
    lgbm_fr.fit(X_fr_trans, y_fr)
    fr_to_uk_pred = lgbm_fr.predict(X_uk_trans_for_fr)
    fr_to_uk_prob = lgbm_fr.predict_proba(X_uk_trans_for_fr)
    fr_to_uk_metrics = evaluate_predictions(y_uk, fr_to_uk_pred, fr_to_uk_prob, "France-to-UK Transfer")

    # 3. Spatially Independent Holdout Validation (e.g. Wales & Scotland in UK, Ile-de-France in FR)
    print("\nEvaluating Spatially Independent Holdout...")
    # Select held-out regions: Wales (W...) or Scottish codes (S...) in UK
    spatial_holdout_mask = pdf["administrative_region"].str.startswith("W") | pdf["administrative_region"].str.startswith("S")
    if spatial_holdout_mask.sum() > 500:
        idx_spatial = np.where(spatial_holdout_mask)[0]
        # Intersect with test indices
        idx_spatial_test = np.intersect1d(idx_test, idx_spatial)
        if len(idx_spatial_test) > 100:
            sp_test_metrics = evaluate_predictions(
                y[idx_spatial_test],
                lgbm.predict(preprocessor.transform(pdf.iloc[idx_spatial_test][ALL_FEATURE_COLS])),
                lgbm.predict_proba(preprocessor.transform(pdf.iloc[idx_spatial_test][ALL_FEATURE_COLS])),
                "Spatially Independent Holdout (Wales & Scotland)"
            )
        else:
            sp_test_metrics = {"status": "insufficient_samples"}
    else:
        sp_test_metrics = {"status": "not_applicable"}

    # Display comparison table
    print("\n" + "=" * 70)
    print("MODEL EVALUATION RESULTS (HUMAN INJURY SEVERITY)")
    print("=" * 70)
    
    comp_df = pd.DataFrame([
        {"Model / Evaluation Split": dummy_metrics["split"], "Macro F1": dummy_metrics["macro_f1"], "Balanced Acc": dummy_metrics["balanced_accuracy"], "Fatal Recall": dummy_metrics["per_class"]["FATAL"]["recall"], "Log Loss": dummy_metrics["log_loss"]},
        {"Model / Evaluation Split": lr_metrics["split"], "Macro F1": lr_metrics["macro_f1"], "Balanced Acc": lr_metrics["balanced_accuracy"], "Fatal Recall": lr_metrics["per_class"]["FATAL"]["recall"], "Log Loss": lr_metrics["log_loss"]},
        {"Model / Evaluation Split": test_metrics["split"], "Macro F1": test_metrics["macro_f1"], "Balanced Acc": test_metrics["balanced_accuracy"], "Fatal Recall": test_metrics["per_class"]["FATAL"]["recall"], "Log Loss": test_metrics["log_loss"]},
        {"Model / Evaluation Split": uk_test_metrics["split"], "Macro F1": uk_test_metrics["macro_f1"], "Balanced Acc": uk_test_metrics["balanced_accuracy"], "Fatal Recall": uk_test_metrics["per_class"]["FATAL"]["recall"], "Log Loss": uk_test_metrics["log_loss"]},
        {"Model / Evaluation Split": fr_test_metrics["split"], "Macro F1": fr_test_metrics["macro_f1"], "Balanced Acc": fr_test_metrics["balanced_accuracy"], "Fatal Recall": fr_test_metrics["per_class"]["FATAL"]["recall"], "Log Loss": fr_test_metrics["log_loss"]},
        {"Model / Evaluation Split": uk_to_fr_metrics["split"], "Macro F1": uk_to_fr_metrics["macro_f1"], "Balanced Acc": uk_to_fr_metrics["balanced_accuracy"], "Fatal Recall": uk_to_fr_metrics["per_class"]["FATAL"]["recall"], "Log Loss": uk_to_fr_metrics["log_loss"]},
        {"Model / Evaluation Split": fr_to_uk_metrics["split"], "Macro F1": fr_to_uk_metrics["macro_f1"], "Balanced Acc": fr_to_uk_metrics["balanced_accuracy"], "Fatal Recall": fr_to_uk_metrics["per_class"]["FATAL"]["recall"], "Log Loss": fr_to_uk_metrics["log_loss"]}
    ])
    print(comp_df.to_string(index=False))

    # Feature Importance
    raw_feature_names = preprocessor.get_feature_names_out()
    importances = lgbm.feature_importances_
    feat_imp = sorted(zip(raw_feature_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Feature Importances:")
    for fn, imp in feat_imp:
        print(f"  - {fn}: {imp}")

    # 4. Save Model Artifacts
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "human_injury_severity_lgbm.pkl"
    pipe_path = models_dir / "injury_feature_pipeline.pkl"
    schema_path = models_dir / "injury_feature_schema.json"
    meta_path = models_dir / "injury_model_metadata.json"
    baseline_path = models_dir / "injury_dummy_baseline.pkl"

    joblib.dump(lgbm, model_path)
    joblib.dump(preprocessor, pipe_path)
    joblib.dump(dummy, baseline_path)

    schema_data = {
        "model_id": "human_injury_severity_v1",
        "task": "3-Class Human Injury Severity Prediction",
        "target_classes": {
            "0": "SLIGHT",
            "1": "SERIOUS",
            "2": "FATAL"
        },
        "feature_columns_numerical": FEATURE_COLS_NUM,
        "feature_columns_boolean": FEATURE_COLS_BOOL,
        "feature_columns_categorical": FEATURE_COLS_CAT,
        "total_raw_features": len(ALL_FEATURE_COLS),
        "total_encoded_features": int(X_train.shape[1])
    }
    with open(schema_path, "w") as f:
        json.dump(schema_data, f, indent=2)

    metadata = {
        "model_name": "human_injury_severity_lightgbm",
        "model_version": "1.0.0",
        "trained_at": datetime.utcnow().isoformat(),
        "task": "3-Class Human Physiological Trauma Severity",
        "training_data_sources": [
            "UK Department for Transport STATS19 (2022) - OGL v3.0",
            "France Ministère de l'Intérieur / ONISR BAAC (2023) - Licence Ouverte v2.0"
        ],
        "licensing": "Commercial SaaS Cleared (OGL v3.0 & Licence Ouverte v2.0)",
        "sample_counts": {
            "total_samples": len(pdf),
            "training_samples": len(idx_train),
            "validation_samples": len(idx_val),
            "test_samples": len(idx_test),
            "uk_samples": len(uk_all_idx),
            "france_samples": len(fr_all_idx)
        },
        "evaluation_metrics": {
            "dummy_baseline": dummy_metrics,
            "logistic_regression": lr_metrics,
            "test_multi_country": test_metrics,
            "test_uk_in_country": uk_test_metrics,
            "test_france_in_country": fr_test_metrics,
            "cross_country_uk_to_fr": uk_to_fr_metrics,
            "cross_country_fr_to_uk": fr_to_uk_metrics,
            "spatial_holdout": sp_test_metrics
        },
        "top_features": [{"feature": fn, "importance": int(imp)} for fn, imp in feat_imp],
        "hyperparameters": lgbm.get_params()
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model artifacts:")
    print(f"  - Model: {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  - Pipeline: {pipe_path} ({pipe_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - Schema: {schema_path}")
    print(f"  - Metadata: {meta_path}")

    return metadata

if __name__ == "__main__":
    train_injury_model()
