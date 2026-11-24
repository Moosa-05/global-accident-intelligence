import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

import json
import joblib
import mlflow
import numpy as np
import lightgbm as lgb
from datetime import datetime
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from backend.ml.config import (
    MODEL_PATH,
    DUMMY_MODEL_PATH,
    PIPELINE_PATH,
    METADATA_PATH,
    SCHEMA_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLUMNS,
    RANDOM_STATE
)
from backend.ml.dataset import get_train_val_test_splits
from backend.ml.pipeline import create_preprocessing_pipeline
from backend.ml.evaluate import evaluate_predictions, find_optimal_threshold

def run_debugging_and_training():
    print("=" * 65)
    print("ACCIDENT-RISK MODEL: TARGETED DEBUGGING & BENCHMARKING PASS")
    print("=" * 65)

    X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits()

    print("\nFitting preprocessing pipeline on training set...")
    pipeline = create_preprocessing_pipeline()
    X_train_proc = pipeline.fit_transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)
    print(f"Features: {X_train_proc.shape[1]} columns")

    results_table = {}

    # 1. Baseline: DummyClassifier
    print("\n[1/4] Training Baseline (DummyClassifier)...")
    dummy = DummyClassifier(strategy='prior', random_state=RANDOM_STATE)
    dummy.fit(X_train_proc, y_train)
    dummy_test_proba = dummy.predict_proba(X_test_proc)[:, 1]
    dummy_metrics = evaluate_predictions(y_test, dummy_test_proba, threshold=0.5)
    results_table["Dummy Baseline"] = dummy_metrics
    print(f"  Dummy: PR-AUC={dummy_metrics['pr_auc']:.4f}, ROC-AUC={dummy_metrics['roc_auc']:.4f}")

    # 2. Linear Model: Logistic Regression (Balanced)
    print("\n[2/4] Training Logistic Regression (Balanced)...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_proc, y_train)
    lr_val_proba = lr.predict_proba(X_val_proc)[:, 1]
    lr_thresh, _ = find_optimal_threshold(y_val, lr_val_proba)
    lr_test_proba = lr.predict_proba(X_test_proc)[:, 1]
    lr_metrics = evaluate_predictions(y_test, lr_test_proba, threshold=lr_thresh)
    results_table["Logistic Regression"] = lr_metrics
    print(f"  Logistic Reg: PR-AUC={lr_metrics['pr_auc']:.4f}, ROC-AUC={lr_metrics['roc_auc']:.4f}, Recall={lr_metrics['recall']:.4f}, Prec={lr_metrics['precision']:.4f}, F1={lr_metrics['f1']:.4f}")

    # 3. Tree Ensemble: Random Forest (Balanced)
    print("\n[3/4] Training Random Forest (Balanced)...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_proc, y_train)
    rf_val_proba = rf.predict_proba(X_val_proc)[:, 1]
    rf_thresh, _ = find_optimal_threshold(y_val, rf_val_proba)
    rf_test_proba = rf.predict_proba(X_test_proc)[:, 1]
    rf_metrics = evaluate_predictions(y_test, rf_test_proba, threshold=rf_thresh)
    results_table["Random Forest"] = rf_metrics
    print(f"  Random Forest: PR-AUC={rf_metrics['pr_auc']:.4f}, ROC-AUC={rf_metrics['roc_auc']:.4f}, Recall={rf_metrics['recall']:.4f}, Prec={rf_metrics['precision']:.4f}, F1={rf_metrics['f1']:.4f}")

    # 4. Gradient Boosting: LightGBM Hyperparameter Search on Validation PR-AUC
    print("\n[4/4] Hyperparameter Optimization for LightGBM on Validation PR-AUC...")
    best_lgb_model = None
    best_lgb_params = None
    best_val_pr_auc = -1.0

    param_grid = [
        {'n_estimators': 30, 'learning_rate': 0.03, 'max_depth': 3, 'num_leaves': 7, 'scale_pos_weight': 1.0},
        {'n_estimators': 30, 'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 15, 'scale_pos_weight': 2.0},
        {'n_estimators': 30, 'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 15, 'scale_pos_weight': 3.0},
        {'n_estimators': 50, 'learning_rate': 0.03, 'max_depth': 3, 'num_leaves': 7, 'scale_pos_weight': 2.0},
        {'n_estimators': 50, 'learning_rate': 0.03, 'max_depth': 4, 'num_leaves': 15, 'scale_pos_weight': 3.0},
        {'n_estimators': 50, 'learning_rate': 0.05, 'max_depth': 4, 'num_leaves': 15, 'scale_pos_weight': 1.0},
        {'n_estimators': 80, 'learning_rate': 0.03, 'max_depth': 3, 'num_leaves': 7, 'scale_pos_weight': 2.0},
    ]

    for p in param_grid:
        clf = lgb.LGBMClassifier(
            **p,
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_samples=30,
            random_state=RANDOM_STATE,
            verbose=-1
        )
        clf.fit(X_train_proc, y_train)
        val_proba = clf.predict_proba(X_val_proc)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_proba)
        val_roc = roc_auc_score(y_val, val_proba)
        
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_lgb_model = clf
            best_lgb_params = p

    print(f"\nBest LightGBM Params (by Val PR-AUC = {best_val_pr_auc:.4f}): {best_lgb_params}")
    
    # Optimize decision threshold on Validation Set using best LightGBM model
    val_proba_best = best_lgb_model.predict_proba(X_val_proc)[:, 1]
    best_thresh, val_best_f1 = find_optimal_threshold(y_val, val_proba_best)
    print(f"Validation Optimal Threshold: {best_thresh:.4f} (Val F1: {val_best_f1:.4f})")

    # Evaluate best LightGBM on untouched Test Set
    test_proba_best = best_lgb_model.predict_proba(X_test_proc)[:, 1]
    lgb_test_metrics = evaluate_predictions(y_test, test_proba_best, threshold=best_thresh)
    results_table["LightGBM (Tuned)"] = lgb_test_metrics

    print("\n" + "=" * 65)
    print("FINAL TEST SET PERFORMANCE COMPARISON (April 2021 Holdout):")
    print("=" * 65)
    header = f"{'Model':<22} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Recall':<8} | {'Prec':<8} | {'F1':<8} | {'BalAcc':<8}"
    print(header)
    print("-" * len(header))
    for name, m in results_table.items():
        print(f"{name:<22} | {m['pr_auc']:<8.4f} | {m['roc_auc']:<8.4f} | {m['recall']:<8.4f} | {m['precision']:<8.4f} | {m['f1']:<8.4f} | {m['balanced_accuracy']:<8.4f}")
    print("=" * 65)
    print(f"LightGBM Confusion Matrix: {lgb_test_metrics['confusion_matrix']}")

    # Feature Importance of best model
    feature_names = pipeline.get_feature_names_out()
    importances = best_lgb_model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:10]
    print("\nTop 10 Feature Importances in Tuned LightGBM:")
    for idx in top_indices:
        print(f"  {feature_names[idx]:35}: {importances[idx]}")

    # Track in MLflow
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("accident-risk-prediction-phase1")
    with mlflow.start_run(run_name="lightgbm_tuned_v2") as run:
        mlflow.log_params(best_lgb_params)
        mlflow.log_param("best_threshold", best_thresh)
        mlflow.log_metric("test_pr_auc", lgb_test_metrics["pr_auc"])
        mlflow.log_metric("test_roc_auc", lgb_test_metrics["roc_auc"])
        mlflow.log_metric("test_recall", lgb_test_metrics["recall"])
        mlflow.log_metric("test_precision", lgb_test_metrics["precision"])
        mlflow.log_metric("test_f1", lgb_test_metrics["f1"])
        mlflow.log_metric("test_balanced_accuracy", lgb_test_metrics["balanced_accuracy"])
        mlflow.log_metric("test_brier_score", lgb_test_metrics["brier_score"])

    # Save artifacts
    print("\nUpdating model artifacts in models/ directory...")
    joblib.dump(best_lgb_model, MODEL_PATH)
    joblib.dump(dummy, DUMMY_MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)

    schema_info = {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_feature_columns": ALL_FEATURE_COLUMNS,
        "target": "label",
        "input_count": len(ALL_FEATURE_COLUMNS)
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema_info, f, indent=2)

    metadata = {
        "model_name": "accident_risk_lightgbm_tuned",
        "model_version": "1.1.0",
        "trained_at": datetime.utcnow().isoformat(),
        "optimal_threshold": best_thresh,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "comparison_results": results_table,
        "test_metrics": lgb_test_metrics,
        "top_features": [
            {"feature": feature_names[idx], "importance": int(importances[idx])}
            for idx in top_indices
        ],
        "hyperparameters": best_lgb_params
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Artifacts successfully updated.")
    return metadata

if __name__ == "__main__":
    run_debugging_and_training()
