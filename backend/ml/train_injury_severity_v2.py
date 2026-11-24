"""Phase 13: Global Human Injury Severity Model v2 — Multi-Country Training and Generalization.
Harmonized cross-country training across UK, France, Germany, and Australia (618,571 collisions).
Executes:
1. Four Leave-One-Country-Out (LOCO) generalization experiments.
2. Head-to-head comparison against deployed v1 model on identical test observations.
3. Multi-model benchmarking (Logistic Regression, LightGBM, CatBoost).
4. Persists the new v2 candidate artifact separately for review.
"""
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
import lightgbm as lgb
from catboost import CatBoostClassifier
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PARQUET_PATH = PROCESSED_DIR / "global_canonical_accidents_4countries.parquet"

# Setup MLflow tracking
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
mlflow.set_experiment("global-human-injury-severity-v2")

SEVERITY_NAMES = ["SLIGHT", "SERIOUS", "FATAL"]

FEATURE_COLS_NUM = [
    "speed_limit_kmh",
    "speed_limit_squared_proxy",
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
    "divided_roadway",
    "has_pedestrian",
    "has_bicycle",
    "has_motorcycle",
    "has_heavy_vehicle",
    "has_car"
]

FEATURE_COLS_CAT = [
    "road_class",
    "weather_condition",
    "light_condition",
    "road_surface_condition"
]

ALL_TRANSFERABLE_FEATURES = FEATURE_COLS_NUM + FEATURE_COLS_BOOL + FEATURE_COLS_CAT

def build_transferable_preprocessor() -> ColumnTransformer:
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, FEATURE_COLS_NUM),
            ("bool", "passthrough", FEATURE_COLS_BOOL),
            ("cat", cat_transformer, FEATURE_COLS_CAT)
        ]
    )

def evaluate_predictions(y_true, y_pred, y_prob, split_label: str) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    try:
        ll = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
    except Exception:
        ll = None
        
    # Multiclass Brier score
    y_oh = np.zeros((len(y_true), 3))
    for i, val in enumerate(y_true):
        if val in [0, 1, 2]:
            y_oh[i, val] = 1.0
    brier = float(np.mean(np.sum((y_prob - y_oh) ** 2, axis=1)))
    
    cr = classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=SEVERITY_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    
    return {
        "split": split_label,
        "sample_count": len(y_true),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "log_loss": round(ll, 4) if ll is not None else None,
        "brier_score": round(brier, 4),
        "per_class": {
            "SLIGHT": {
                "precision": round(cr["SLIGHT"]["precision"], 4),
                "recall": round(cr["SLIGHT"]["recall"], 4),
                "f1": round(cr["SLIGHT"]["f1-score"], 4),
                "support": int(cr["SLIGHT"]["support"])
            },
            "SERIOUS": {
                "precision": round(cr["SERIOUS"]["precision"], 4),
                "recall": round(cr["SERIOUS"]["recall"], 4),
                "f1": round(cr["SERIOUS"]["f1-score"], 4),
                "support": int(cr["SERIOUS"]["support"])
            },
            "FATAL": {
                "precision": round(cr["FATAL"]["precision"], 4),
                "recall": round(cr["FATAL"]["recall"], 4),
                "f1": round(cr["FATAL"]["f1-score"], 4),
                "support": int(cr["FATAL"]["support"])
            }
        },
        "confusion_matrix": cm
    }

def print_metrics_summary(res: dict, title: str):
    print(f"\n{title} (n = {res['sample_count']:,}):")
    print(f"  Macro F1:          {res['macro_f1']:.4f}")
    print(f"  Weighted F1:       {res['weighted_f1']:.4f}")
    print(f"  Balanced Accuracy: {res['balanced_accuracy']:.4f} ({res['balanced_accuracy']:.2%})")
    print(f"  Log Loss:          {res['log_loss']}")
    print(f"  Brier Score:       {res['brier_score']:.4f}")
    print("  Per-Class Metrics:")
    for cls_name in SEVERITY_NAMES:
        m = res["per_class"][cls_name]
        print(f"    {cls_name:<10}: Precision={m['precision']:.4f} | Recall={m['recall']:.4f} | F1={m['f1']:.4f} | Support={m['support']:,}")
    print("  Confusion Matrix (Rows=True, Cols=Pred):")
    cm = res["confusion_matrix"]
    print(f"    {'':<10} {'Pred SLIGHT':<12} {'Pred SERIOUS':<14} {'Pred FATAL':<12}")
    for idx, name in enumerate(SEVERITY_NAMES):
        print(f"    True {name:<6}: {cm[idx][0]:<12} {cm[idx][1]:<14} {cm[idx][2]:<12}")

def main():
    print("=" * 80)
    print("PHASE 13: GLOBAL HUMAN INJURY SEVERITY MODEL v2 PIPELINE")
    print("=" * 80)

    # 1. Load Harmonized 4-Country Dataset
    print("\n[1/6] Loading Harmonized Multi-Country Parquet...")
    t0 = time.time()
    df_pl = pl.read_parquet(PARQUET_PATH)
    pdf = df_pl.to_pandas()
    print(f"  Loaded {len(pdf):,} collision records in {time.time()-t0:.2f}s.")
    print("  Country Counts:")
    for ccode, count in pdf["country_code"].value_counts().items():
        print(f"    {ccode}: {count:,}")

    # Ensure boolean columns are strictly float 0.0/1.0 without NaNs
    for c in FEATURE_COLS_BOOL:
        pdf[c] = pdf[c].fillna(False).astype(float)

    # Build Preprocessor (Strictly Coordinate-Free & Country-Invariant)
    preprocessor = build_transferable_preprocessor()

    # Load existing v1 model and pipeline for direct head-to-head comparison
    v1_model = joblib.load(MODELS_DIR / "human_injury_severity_lgbm.pkl")
    v1_pipe = joblib.load(MODELS_DIR / "injury_feature_pipeline.pkl")
    
    # Existing v1 feature columns
    v1_num = ["latitude", "longitude", "speed_limit_kmh", "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_kmh", "precipitation_mm"]
    v1_bool = ["is_junction", "is_traffic_signal", "is_crossing", "is_roundabout", "is_stop_sign", "is_give_way", "is_railway", "is_weekend", "is_night", "divided_roadway"]
    v1_cat = ["road_class", "weather_condition", "light_condition", "road_surface_condition"]
    v1_all_cols = v1_num + v1_bool + v1_cat

    # -------------------------------------------------------------
    # 2. FOUR LEAVE-ONE-COUNTRY-OUT (LOCO) EXPERIMENTS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[2/6] EXECUTING FOUR LEAVE-ONE-COUNTRY-OUT (LOCO) GENERALIZATION EXPERIMENTS")
    print("=" * 80)

    loco_results = {}
    countries = ["AU", "DE", "FR", "GB"]
    country_names = {
        "AU": "Australia (Transport Victoria)",
        "DE": "Germany (Destatis Unfallatlas)",
        "FR": "France (ONISR BAAC)",
        "GB": "United Kingdom (DfT STATS19)"
    }

    for test_country in countries:
        train_countries = [c for c in countries if c != test_country]
        print(f"\n" + "-" * 75)
        print(f"LOCO EXPERIMENT: Test on {country_names[test_country]} [{test_country}]")
        print(f"                 Train on {', '.join(train_countries)}")
        print("-" * 75)

        train_mask = pdf["country_code"].isin(train_countries).values
        test_mask = (pdf["country_code"] == test_country).values

        df_tr = pdf[train_mask]
        df_te = pdf[test_mask]

        # Fit preprocessor strictly on training countries
        X_tr = preprocessor.fit_transform(df_tr[ALL_TRANSFERABLE_FEATURES])
        X_te = preprocessor.transform(df_te[ALL_TRANSFERABLE_FEATURES])
        y_tr = df_tr["target"].values
        y_te = df_te["target"].values

        print(f"  Training Samples ({', '.join(train_countries)}): {len(y_tr):,}")
        print(f"  Test Samples ({test_country}):                      {len(y_te):,}")

        # Train Transferable LightGBM LOCO Model
        lgb_loco = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            learning_rate=0.06,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.80,
            n_estimators=120,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_loco.fit(X_tr, y_tr)

        # Predict with Transferable v2 LOCO Model
        pred_v2 = lgb_loco.predict(X_te)
        prob_v2 = lgb_loco.predict_proba(X_te)
        res_v2 = evaluate_predictions(y_te, pred_v2, prob_v2, f"v2 LOCO Test ({test_country})")

        # Evaluate existing v1 model on identical test observations
        X_te_v1 = v1_pipe.transform(df_te[v1_all_cols])
        pred_v1 = v1_model.predict(X_te_v1)
        prob_v1 = v1_model.predict_proba(X_te_v1)
        res_v1 = evaluate_predictions(y_te, pred_v1, prob_v1, f"Existing v1 Test ({test_country})")

        print_metrics_summary(res_v1, f"Existing v1 Model on Unseen {test_country}")
        print_metrics_summary(res_v2, f"Transferable v2 LOCO Model on Unseen {test_country}")

        macro_delta = res_v2["macro_f1"] - res_v1["macro_f1"]
        bal_delta = res_v2["balanced_accuracy"] - res_v1["balanced_accuracy"]
        print(f"\n  >>> LOCO Improvement on {test_country}: Macro F1 delta = {macro_delta:+.4f} | Balanced Acc delta = {bal_delta:+.2%}")

        loco_results[test_country] = {
            "v1": res_v1,
            "v2": res_v2,
            "macro_f1_delta": round(macro_delta, 4),
            "balanced_acc_delta": round(bal_delta, 4)
        }

    # -------------------------------------------------------------
    # 3. POOLED MULTI-COUNTRY TRAINING FOR GLOBAL CANDIDATE v2
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3/6] POOLED MULTI-COUNTRY TRAINING (UK + FR + DE + AU)")
    print("=" * 80)

    # Stratified 80% Train, 10% Val, 10% Test split across all 4 countries
    indices = np.arange(len(pdf))
    y_all = pdf["target"].values
    country_all = pdf["country_code"].values

    # Stratify by both target and country_code
    strata = pdf["country_code"] + "_" + pdf["target"].astype(str)
    
    idx_tr, idx_temp, y_tr, y_temp = train_test_split(indices, y_all, test_size=0.20, random_state=42, stratify=strata)
    strata_temp = strata.iloc[idx_temp]
    idx_va, idx_te, y_va, y_te = train_test_split(idx_temp, y_temp, test_size=0.50, random_state=42, stratify=strata_temp)

    print(f"  Training Samples (80%):   {len(idx_tr):,}")
    print(f"  Validation Samples (10%): {len(idx_va):,}")
    print(f"  Test Samples (10%):       {len(idx_te):,}")

    # Fit feature preprocessor on full 80% multi-country training split
    preprocessor_v2 = build_transferable_preprocessor()
    X_tr_full = preprocessor_v2.fit_transform(pdf.iloc[idx_tr][ALL_TRANSFERABLE_FEATURES])
    X_va_full = preprocessor_v2.transform(pdf.iloc[idx_va][ALL_TRANSFERABLE_FEATURES])
    X_te_full = preprocessor_v2.transform(pdf.iloc[idx_te][ALL_TRANSFERABLE_FEATURES])

    print(f"  Transformed Feature Dimension: {X_tr_full.shape[1]} columns")

    models_benchmark = {}

    # Model 1: Regularized Logistic Regression Baseline
    print("\n  [A] Training Logistic Regression Baseline (Multinomial Balanced)...")
    lr = LogisticRegression(class_weight="balanced", max_iter=250, random_state=42, solver="lbfgs")
    lr.fit(X_tr_full, y_tr)
    lr_pred = lr.predict(X_te_full)
    lr_prob = lr.predict_proba(X_te_full)
    res_lr = evaluate_predictions(y_te, lr_pred, lr_prob, "Logistic Regression Baseline")
    models_benchmark["Logistic Regression"] = res_lr
    print(f"      Macro F1 = {res_lr['macro_f1']:.4f} | Balanced Acc = {res_lr['balanced_accuracy']:.2%} | Fatal Recall = {res_lr['per_class']['FATAL']['recall']:.2%}")

    # Model 2: LightGBM Multiclass (Balanced)
    print("\n  [B] Training LightGBM Multiclass Classifier...")
    lgbm_v2 = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight="balanced",
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.80,
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_v2.fit(X_tr_full, y_tr, eval_set=[(X_va_full, y_va)], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
    lgbm_pred = lgbm_v2.predict(X_te_full)
    lgbm_prob = lgbm_v2.predict_proba(X_te_full)
    res_lgbm = evaluate_predictions(y_te, lgbm_pred, lgbm_prob, "LightGBM Multiclass")
    models_benchmark["LightGBM"] = res_lgbm
    print(f"      Macro F1 = {res_lgbm['macro_f1']:.4f} | Balanced Acc = {res_lgbm['balanced_accuracy']:.2%} | Fatal Recall = {res_lgbm['per_class']['FATAL']['recall']:.2%}")

    # Model 3: CatBoost Multiclass
    print("\n  [C] Training CatBoost Multiclass Classifier...")
    cat_v2 = CatBoostClassifier(
        loss_function="MultiClass",
        auto_class_weights="Balanced",
        iterations=150,
        learning_rate=0.08,
        depth=6,
        random_seed=42,
        verbose=False,
        thread_count=-1
    )
    cat_v2.fit(X_tr_full, y_tr, eval_set=(X_va_full, y_va), early_stopping_rounds=15)
    cat_pred = cat_v2.predict(X_te_full).flatten()
    cat_prob = cat_v2.predict_proba(X_te_full)
    res_cat = evaluate_predictions(y_te, cat_pred, cat_prob, "CatBoost Multiclass")
    models_benchmark["CatBoost"] = res_cat
    print(f"      Macro F1 = {res_cat['macro_f1']:.4f} | Balanced Acc = {res_cat['balanced_accuracy']:.2%} | Fatal Recall = {res_cat['per_class']['FATAL']['recall']:.2%}")

    # Model 4: Calibrated Weighted LightGBM (Safety-Critical Trauma Optimized)
    print("\n  [D] Training Trauma-Calibrated LightGBM (Custom Safety Weights 1.0 : 2.6 : 14.0)...")
    safety_weights = {0: 1.0, 1: 2.6, 2: 14.0}
    lgbm_safety = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        class_weight=safety_weights,
        learning_rate=0.06,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.80,
        n_estimators=150,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm_safety.fit(X_tr_full, y_tr, eval_set=[(X_va_full, y_va)], callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)])
    safety_pred = lgbm_safety.predict(X_te_full)
    safety_prob = lgbm_safety.predict_proba(X_te_full)
    res_safety = evaluate_predictions(y_te, safety_pred, safety_prob, "Trauma-Calibrated LightGBM Candidate")
    models_benchmark["Trauma-Calibrated LightGBM"] = res_safety
    print_metrics_summary(res_safety, "Trauma-Calibrated LightGBM Global Candidate")

    # -------------------------------------------------------------
    # 4. FEATURE ATTRIBUTION / IMPORTANCE ANALYSIS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[4/6] FEATURE IMPORTANCE & ATTRIBUTION ANALYSIS (GLOBAL CANDIDATE v2)")
    print("=" * 80)
    
    # Extract feature names after OneHotEncoder
    cat_encoder = preprocessor_v2.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = list(cat_encoder.get_feature_names_out(FEATURE_COLS_CAT))
    all_feature_names = FEATURE_COLS_NUM + FEATURE_COLS_BOOL + cat_feature_names

    importances = lgbm_safety.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    print("Top 15 Most Influential Features (Gain / Split Importance):")
    for i in range(min(15, len(all_feature_names))):
        f_idx = sorted_idx[i]
        print(f"  {i+1:>2}. {all_feature_names[f_idx]:<35}: {importances[f_idx]:>6}")

    # -------------------------------------------------------------
    # 5. PERSIST CANDIDATE ARTIFACTS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[5/6] PERSISTING CANDIDATE ARTIFACTS")
    print("=" * 80)

    candidate_model_path = MODELS_DIR / "human_injury_severity_global_v2_candidate.pkl"
    candidate_pipe_path = MODELS_DIR / "injury_feature_pipeline_v2.pkl"
    candidate_meta_path = MODELS_DIR / "injury_model_v2_metadata.json"

    # Save candidate model & pipeline
    joblib.dump(lgbm_safety, candidate_model_path)
    joblib.dump(preprocessor_v2, candidate_pipe_path)
    print(f"  Saved candidate model to: {candidate_model_path.name}")
    print(f"  Saved candidate preprocessor to: {candidate_pipe_path.name}")

    candidate_metadata = {
        "model_name": "human_injury_severity_global_v2_candidate",
        "model_version": "2.0.0-candidate",
        "trained_at": datetime.utcnow().isoformat(),
        "task": "3-Class Physiological Trauma Severity (SLIGHT, SERIOUS, FATAL)",
        "training_data_sources": [
            "United Kingdom DfT STATS19 (2022) - OGL v3.0",
            "France ONISR BAAC (2023) - Licence Ouverte v2.0",
            "Germany Destatis Unfallatlas (2022) - dl-de/by-2-0",
            "Australia Transport Victoria Crash Data (2012-2022) - CC BY 4.0"
        ],
        "total_records_harmonized": len(pdf),
        "loco_cross_country_evaluation": loco_results,
        "pooled_test_metrics_10pct": res_safety,
        "models_benchmark_comparison": models_benchmark,
        "feature_columns": {
            "numerical": FEATURE_COLS_NUM,
            "boolean": FEATURE_COLS_BOOL,
            "categorical": FEATURE_COLS_CAT,
            "excluded_for_invariance": ["latitude", "longitude", "country_code"]
        }
    }
    with open(candidate_meta_path, "w") as f:
        json.dump(candidate_metadata, f, indent=2)
    print(f"  Saved candidate metadata to: {candidate_meta_path.name}")

    # Log to MLflow
    try:
        with mlflow.start_run(run_name="human_injury_severity_v2_global_candidate"):
            mlflow.log_params({
                "n_samples": len(pdf),
                "countries": "GB,FR,DE,AU",
                "model_type": "LightGBM",
                "class_weights": "1.0:2.6:14.0",
                "learning_rate": 0.06,
                "num_leaves": 31
            })
            mlflow.log_metrics({
                "test_macro_f1": res_safety["macro_f1"],
                "test_balanced_acc": res_safety["balanced_accuracy"],
                "test_log_loss": res_safety["log_loss"],
                "test_brier_score": res_safety["brier_score"],
                "fatal_recall": res_safety["per_class"]["FATAL"]["recall"],
                "fatal_precision": res_safety["per_class"]["FATAL"]["precision"],
                "serious_recall": res_safety["per_class"]["SERIOUS"]["recall"],
                "serious_precision": res_safety["per_class"]["SERIOUS"]["precision"],
                "loco_au_macro_f1": loco_results["AU"]["v2"]["macro_f1"],
                "loco_de_macro_f1": loco_results["DE"]["v2"]["macro_f1"],
                "loco_fr_macro_f1": loco_results["FR"]["v2"]["macro_f1"],
                "loco_gb_macro_f1": loco_results["GB"]["v2"]["macro_f1"]
            })
            print("  Logged experiment metrics to MLflow.")
    except Exception as e:
        print(f"  MLflow logging note: {e}")

    print("\n" + "=" * 80)
    print("PHASE 13 EXECUTION & VALIDATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
