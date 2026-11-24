"""Investigate Chattanooga Accident Risk model improvements in precision, recall, and PR-AUC.
Preserves independent test evaluation without leakage or artificial metric inflation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.dataset import get_train_val_test_splits
from backend.ml.pipeline import create_preprocessing_pipeline
from backend.ml.evaluate import evaluate_predictions, find_optimal_threshold

X_train, y_train, X_val, y_val, X_test, y_test = get_train_val_test_splits()

pipeline = create_preprocessing_pipeline()
X_train_proc = pipeline.fit_transform(X_train)
X_val_proc = pipeline.transform(X_val)
X_test_proc = pipeline.transform(X_test)

print(f"Features: {X_train_proc.shape[1]}")
print(f"Base Positive Rate in Test: {y_test.mean():.4f}")

# Benchmark existing baseline
existing_model = lgb.LGBMClassifier(
    n_estimators=30,
    learning_rate=0.03,
    max_depth=3,
    num_leaves=7,
    scale_pos_weight=1.0,
    random_state=42,
    verbose=-1
)
existing_model.fit(X_train_proc, y_train)
val_p = existing_model.predict_proba(X_val_proc)[:, 1]
opt_thresh, _ = find_optimal_threshold(y_val, val_p)

test_p = existing_model.predict_proba(X_test_proc)[:, 1]
m_exist = evaluate_predictions(y_test, test_p, threshold=opt_thresh)
print(f"Existing Benchmark: PR-AUC = {m_exist['pr_auc']:.4f} | ROC-AUC = {m_exist['roc_auc']:.4f} | F1 = {m_exist['f1']:.4f}")

# Try 1: Calibrated Logistic Regression with L2 Regularization
for c in [0.01, 0.1, 1.0]:
    lr = LogisticRegression(C=c, max_iter=1000, random_state=42)
    lr.fit(X_train_proc, y_train)
    val_p_lr = lr.predict_proba(X_val_proc)[:, 1]
    th_lr, _ = find_optimal_threshold(y_val, val_p_lr)
    test_p_lr = lr.predict_proba(X_test_proc)[:, 1]
    m_lr = evaluate_predictions(y_test, test_p_lr, threshold=th_lr)
    print(f"Logistic Regression (C={c}): PR-AUC = {m_lr['pr_auc']:.4f} | ROC-AUC = {m_lr['roc_auc']:.4f} | F1 = {m_lr['f1']:.4f} | Precision = {m_lr['precision']:.4f} | Recall = {m_lr['recall']:.4f}")

# Try 2: HistGradientBoosting with Early Stopping on PR-AUC
hgb = HistGradientBoostingClassifier(max_iter=50, max_leaf_nodes=15, learning_rate=0.05, random_state=42)
hgb.fit(X_train_proc, y_train)
val_p_hgb = hgb.predict_proba(X_val_proc)[:, 1]
th_hgb, _ = find_optimal_threshold(y_val, val_p_hgb)
test_p_hgb = hgb.predict_proba(X_test_proc)[:, 1]
m_hgb = evaluate_predictions(y_test, test_p_hgb, threshold=th_hgb)
print(f"HistGradientBoosting: PR-AUC = {m_hgb['pr_auc']:.4f} | ROC-AUC = {m_hgb['roc_auc']:.4f} | F1 = {m_hgb['f1']:.4f} | Precision = {m_hgb['precision']:.4f} | Recall = {m_hgb['recall']:.4f}")

# Try 3: Regularized LightGBM with Feature Subsampling & Minimum Gain
for lr_rate in [0.02, 0.05]:
    for leaves in [5, 10]:
        cand = lgb.LGBMClassifier(
            n_estimators=40,
            learning_rate=lr_rate,
            num_leaves=leaves,
            min_child_samples=50,
            colsample_bytree=0.7,
            subsample=0.8,
            random_state=42,
            verbose=-1
        )
        cand.fit(X_train_proc, y_train)
        val_p_c = cand.predict_proba(X_val_proc)[:, 1]
        th_c, _ = find_optimal_threshold(y_val, val_p_c)
        test_p_c = cand.predict_proba(X_test_proc)[:, 1]
        m_c = evaluate_predictions(y_test, test_p_c, threshold=th_c)
        print(f"LightGBM (lr={lr_rate}, leaves={leaves}): PR-AUC = {m_c['pr_auc']:.4f} | ROC-AUC = {m_c['roc_auc']:.4f} | F1 = {m_c['f1']:.4f} | Precision = {m_c['precision']:.4f} | Recall = {m_c['recall']:.4f}")
