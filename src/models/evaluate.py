"""
Model evaluation utilities for fraud detection.
"""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import (
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_classifier(
    y_true: NDArray | pd.Series,
    y_pred: NDArray,
    y_proba: NDArray,
    pos_label: int = 1,  # 1=illicit in Elliptic; pass 2 if using raw class labels
) -> dict[str, float]:
    """Returns f1_weighted, f1_fraud, precision, recall, roc_auc as a dict."""
    metrics = {
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_fraud": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    return metrics


def optimize_threshold(
    y_true: NDArray | pd.Series,
    y_proba: NDArray,
    pos_label: int = 1,
) -> tuple[float, NDArray, dict[str, float]]:
    """Find the threshold that maximises fraud F1.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted probability scores for the positive class.
        pos_label: Label value considered as the positive (fraud) class.

    Returns:
        optimal_threshold: Threshold value that maximises fraud F1.
        y_pred_opt: Hard predictions using the optimal threshold.
        metrics_opt: Evaluation metrics dict at the optimal threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=pos_label)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx = int(np.argmax(f1_scores))
    optimal_threshold = float(thresholds[optimal_idx])

    y_pred_opt = (
        (y_proba > optimal_threshold).astype(int)
        if pos_label == 1
        else np.where(y_proba > optimal_threshold, 1, 2)
    )
    metrics_opt = evaluate_classifier(y_true, y_pred_opt, y_proba, pos_label=pos_label)

    logger.info(
        "Optimal threshold: %.4f → F1=%.4f, Precision=%.4f, Recall=%.4f",
        optimal_threshold,
        metrics_opt["f1_fraud"],
        metrics_opt["precision"],
        metrics_opt["recall"],
    )

    return optimal_threshold, y_pred_opt, metrics_opt


def optimize_threshold_cost_sensitive(
    y_true: NDArray | pd.Series,
    y_proba: NDArray,
    fraud_value: float = 10_000.0,
    investigation_cost: float = 50.0,
    pos_label: int = 1,
) -> tuple[float, float, NDArray, dict[str, float]]:
    """Find the threshold that maximises expected net business value.

    Each caught fraud saves fraud_value; each alert costs investigation_cost
    whether it's a true or false positive; each missed fraud loses fraud_value.
    So: net_value = TP * fraud_value - (TP + FP) * investigation_cost - FN * fraud_value

    Returns (f1_threshold, cost_threshold, y_pred_cost, metrics_cost) where
    metrics_cost includes net_value and net_value_per_case in addition to the
    standard evaluate_classifier keys.

    Raises:
        ValueError: If fraud_value or investigation_cost is negative.
    """
    if fraud_value < 0:
        raise ValueError(f"fraud_value must be non-negative, got {fraud_value}")
    if investigation_cost < 0:
        raise ValueError(f"investigation_cost must be non-negative, got {investigation_cost}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba, pos_label=pos_label)

    # F1-optimal threshold (same as optimize_threshold — kept for comparison)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    f1_optimal_threshold = float(thresholds[int(np.argmax(f1_scores))])

    # Compute confusion matrix components at each threshold
    n_positive = int((np.asarray(y_true) == pos_label).sum())
    n_total = len(y_true)

    tp_arr = recall[:-1] * n_positive
    fp_arr = tp_arr / (precision[:-1] + 1e-10) - tp_arr  # FP = TP/precision − TP
    fn_arr = n_positive - tp_arr

    net_value = tp_arr * fraud_value - (tp_arr + fp_arr) * investigation_cost - fn_arr * fraud_value
    cost_optimal_idx = int(np.argmax(net_value))
    cost_optimal_threshold = float(thresholds[cost_optimal_idx])

    y_pred_cost = (y_proba >= cost_optimal_threshold).astype(int)
    metrics_cost = evaluate_classifier(y_true, y_pred_cost, y_proba, pos_label=pos_label)
    metrics_cost["net_value"] = float(net_value[cost_optimal_idx])
    metrics_cost["net_value_per_case"] = float(net_value[cost_optimal_idx]) / max(n_total, 1)

    logger.info(
        "Cost-sensitive optimisation (fraud_value=%.0f, investigation_cost=%.0f): "
        "F1-optimal=%.4f, cost-optimal=%.4f → "
        "F1=%.4f, Net value=%.0f, Net value/case=%.2f",
        fraud_value,
        investigation_cost,
        f1_optimal_threshold,
        cost_optimal_threshold,
        metrics_cost["f1_fraud"],
        metrics_cost["net_value"],
        metrics_cost["net_value_per_case"],
    )

    return f1_optimal_threshold, cost_optimal_threshold, y_pred_cost, metrics_cost
