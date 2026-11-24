import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes F1 score on validation predictions.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Compute F1 scores for each threshold (avoid divide-by-zero)
    f1_scores = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        p, r = precisions[i], recalls[i]
        if (p + r) > 0:
            f1_scores[i] = 2 * (p * r) / (p + r)
        else:
            f1_scores[i] = 0.0

    best_idx = np.argmax(f1_scores) if len(f1_scores) > 0 else 0
    best_thresh = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
    best_f1 = float(f1_scores[best_idx]) if len(f1_scores) > 0 else 0.0
    return best_thresh, best_f1

def evaluate_predictions(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Calculates complete binary classification metrics including PR-AUC,
    recall, precision, F1, balanced accuracy, confusion matrix, and calibration.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    pr_auc = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.0
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    brier = float(brier_score_loss(y_true, y_proba))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    # Calibration curve (5 bins)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=5, strategy='uniform')

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": round(threshold, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "support": {
            "total_samples": len(y_true),
            "positive_accidents": int(np.sum(y_true)),
            "negative_non_accidents": int(len(y_true) - np.sum(y_true))
        },
        "calibration": {
            "prob_true": [round(float(x), 4) for x in prob_true],
            "prob_pred": [round(float(x), 4) for x in prob_pred]
        }
    }
