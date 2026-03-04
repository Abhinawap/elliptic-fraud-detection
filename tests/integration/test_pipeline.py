"""
Integration test: full pipeline from raw CSV → train XGBoost → verify F1 ≥ 0.65.

Skipped automatically when ELLIPTIC_DATA_DIR environment variable is not set,
so CI passes without the 200MB+ real dataset. To run locally:

    ELLIPTIC_DATA_DIR=/path/to/data/raw pytest tests/integration/ -v
"""

import os
from pathlib import Path

import pytest
import xgboost as xgb

DATA_DIR = os.environ.get("ELLIPTIC_DATA_DIR")

pytestmark = pytest.mark.skipif(
    DATA_DIR is None,
    reason="ELLIPTIC_DATA_DIR not set — skipping integration tests",
)


@pytest.fixture(scope="module")
def pipeline_artifacts():
    """Run the full training pipeline once and return results for all tests."""
    from src.data.loader import load_elliptic_data, merge_and_filter_labeled
    from src.features.feature_engineering import add_missingness_indicator, compute_graph_features
    from src.models.evaluate import evaluate_classifier, optimize_threshold
    from src.models.train import train_classifier
    from src.preprocessing.pipeline import (
        create_preprocessing_pipeline,
        prepare_features_and_labels,
        temporal_train_test_split,
    )
    from src.utils.config import XGB_PARAMS

    data_dir = Path(DATA_DIR)

    # 1. Load
    classes, features, edgelist, has_graph = load_elliptic_data(data_dir)
    _, labeled = merge_and_filter_labeled(features, classes)

    # 2. Feature engineering
    labeled = add_missingness_indicator(labeled)
    if has_graph and edgelist is not None:
        labeled = compute_graph_features(labeled, edgelist)

    # 3. Prepare and split
    X, y, time_steps, feature_cols = prepare_features_and_labels(labeled)
    X_train, X_test, y_train, y_test, _, _ = temporal_train_test_split(X, y, time_steps)

    # 4. Preprocess
    pipeline = create_preprocessing_pipeline()
    X_train_np = pipeline.fit_transform(X_train)
    X_test_np = pipeline.transform(X_test)

    # 5. Convert labels: illicit(1)→1, licit(2)→0
    y_train_bin = (y_train == 1).astype(int).values
    y_test_bin = (y_test == 1).astype(int).values

    # 6. Train
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model, y_pred, y_proba = train_classifier(
        model, X_train_np, y_train_bin, X_test_np, y_test_bin, "XGB-integration"
    )

    # 7. Evaluate
    metrics_default = evaluate_classifier(y_test_bin, y_pred, y_proba, pos_label=1)
    optimal_threshold, y_pred_opt, metrics_opt = optimize_threshold(y_test_bin, y_proba, pos_label=1)

    return {
        "metrics_default": metrics_default,
        "metrics_opt": metrics_opt,
        "optimal_threshold": optimal_threshold,
        "feature_cols": feature_cols,
        "X_train": X_train_np,
        "X_test": X_test_np,
        "y_test": y_test_bin,
    }


class TestPipelineIntegration:
    """End-to-end pipeline integration tests (require ELLIPTIC_DATA_DIR)."""

    def test_fraud_f1_meets_bar(self, pipeline_artifacts):
        """F1 score at optimal threshold must be ≥ 0.65."""
        f1 = pipeline_artifacts["metrics_opt"]["f1_fraud"]
        assert f1 >= 0.65, f"Fraud F1 {f1:.4f} below minimum bar of 0.65"

    def test_roc_auc_meets_bar(self, pipeline_artifacts):
        """ROC-AUC must be ≥ 0.90 (strong discrimination required)."""
        auc = pipeline_artifacts["metrics_opt"]["roc_auc"]
        assert auc >= 0.90, f"ROC-AUC {auc:.4f} below minimum bar of 0.90"

    def test_optimal_threshold_in_valid_range(self, pipeline_artifacts):
        """Optimal threshold must be a valid probability value."""
        thr = pipeline_artifacts["optimal_threshold"]
        assert 0.0 < thr < 1.0, f"Optimal threshold {thr} outside (0, 1)"

    def test_features_include_engineered_cols(self, pipeline_artifacts):
        """Pipeline must include missingness indicator and graph degree features."""
        feature_cols = pipeline_artifacts["feature_cols"]
        assert "is_structural_missing" in feature_cols, "is_structural_missing not in features"
        assert "graph_total_degree" in feature_cols, "graph_total_degree not in features"

    def test_train_test_shapes_consistent(self, pipeline_artifacts):
        """Train and test feature arrays must have the same number of columns."""
        n_train_cols = pipeline_artifacts["X_train"].shape[1]
        n_test_cols = pipeline_artifacts["X_test"].shape[1]
        assert n_train_cols == n_test_cols, (
            f"Feature count mismatch: train={n_train_cols}, test={n_test_cols}"
        )

    def test_precision_not_degenerate(self, pipeline_artifacts):
        """Precision must be > 0 (model predicts at least some positives correctly)."""
        precision = pipeline_artifacts["metrics_opt"]["precision"]
        assert precision > 0.0, "Precision is 0 — model may be predicting no positives"

    def test_recall_not_degenerate(self, pipeline_artifacts):
        """Recall must be > 0 (model catches at least some fraud)."""
        recall = pipeline_artifacts["metrics_opt"]["recall"]
        assert recall > 0.0, "Recall is 0 — model may be missing all fraud cases"
