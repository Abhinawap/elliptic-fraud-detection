"""Unit tests for src/models/train.py and src/models/evaluate.py"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="module")
def small_train_test(synthetic_labeled):
    """Preprocessed train/test split using the synthetic labeled dataset."""
    from src.preprocessing.pipeline import (
        create_preprocessing_pipeline,
        prepare_features_and_labels,
        temporal_train_test_split,
    )

    X, y, time_steps, _ = prepare_features_and_labels(synthetic_labeled)
    X_train, X_test, y_train, y_test, _, _ = temporal_train_test_split(
        X, y, time_steps, split_ratio=0.8
    )
    pipe = create_preprocessing_pipeline()
    X_train_np = pipe.fit_transform(X_train)
    X_test_np = pipe.transform(X_test)
    return X_train_np, X_test_np, y_train.values, y_test.values


class TestTrainClassifier:
    def test_returns_three_items(self, small_train_test):
        from src.models.train import train_classifier

        X_train, X_test, y_train, y_test = small_train_test
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        result = train_classifier(model, X_train, y_train, X_test, y_test, "test_rf")
        assert len(result) == 3

    def test_predictions_correct_length(self, small_train_test):
        from src.models.train import train_classifier

        X_train, X_test, y_train, y_test = small_train_test
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        _, y_pred, y_proba = train_classifier(model, X_train, y_train, X_test, y_test, "test_rf")
        assert len(y_pred) == len(y_test)
        assert len(y_proba) == len(y_test)

    def test_probabilities_in_unit_interval(self, small_train_test):
        from src.models.train import train_classifier

        X_train, X_test, y_train, y_test = small_train_test
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        _, _, y_proba = train_classifier(model, X_train, y_train, X_test, y_test, "rf")
        assert (y_proba >= 0).all()
        assert (y_proba <= 1).all()

    def test_model_is_fitted(self, small_train_test):
        from sklearn.utils.validation import check_is_fitted
        from src.models.train import train_classifier

        X_train, X_test, y_train, y_test = small_train_test
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        fitted_model, _, _ = train_classifier(model, X_train, y_train, X_test, y_test, "rf")
        # Should not raise — model must be fitted
        check_is_fitted(fitted_model)


class TestEvaluateClassifier:
    def test_returns_expected_metric_keys(self, small_train_test):
        from src.models.evaluate import evaluate_classifier

        _, X_test, _, y_test = small_train_test
        rng = np.random.RandomState(0)
        y_pred = rng.choice([1, 2], size=len(y_test))
        y_proba = rng.rand(len(y_test))
        metrics = evaluate_classifier(y_test, y_pred, y_proba, pos_label=1)
        assert set(metrics.keys()) == {
            "f1_weighted",
            "f1_fraud",
            "precision",
            "recall",
            "roc_auc",
        }

    def test_metric_values_in_valid_range(self, small_train_test):
        from src.models.evaluate import evaluate_classifier

        _, X_test, _, y_test = small_train_test
        rng = np.random.RandomState(0)
        y_pred = rng.choice([1, 2], size=len(y_test))
        y_proba = rng.rand(len(y_test))
        metrics = evaluate_classifier(y_test, y_pred, y_proba, pos_label=1)
        for val in metrics.values():
            assert 0.0 <= val <= 1.0, f"Metric out of range: {val}"

    def test_perfect_predictions_give_high_f1(self, small_train_test):
        from src.models.evaluate import evaluate_classifier

        _, _, _, y_test = small_train_test
        # Use binary labels (0=licit, 1=illicit) so roc_auc_score interprets
        # probabilities correctly: high proba → label 1 (illicit).
        y_test_bin = (y_test == 1).astype(int)
        y_pred = y_test_bin.copy()
        y_proba = y_test_bin.astype(float)
        metrics = evaluate_classifier(y_test_bin, y_pred, y_proba, pos_label=1)
        assert metrics["f1_weighted"] > 0.9
        assert metrics["roc_auc"] > 0.9


class TestOptimizeThreshold:
    def test_optimal_threshold_in_unit_interval(self, small_train_test):
        from src.models.evaluate import optimize_threshold

        _, _, _, y_test = small_train_test
        # optimize_threshold with pos_label=1 produces binary {0,1} predictions,
        # so y_true must also be binary.
        y_test_bin = (y_test == 1).astype(int)
        rng = np.random.RandomState(42)
        y_proba = rng.rand(len(y_test_bin))
        threshold, _, _ = optimize_threshold(y_test_bin, y_proba, pos_label=1)
        assert 0.0 <= threshold <= 1.0

    def test_returns_three_items(self, small_train_test):
        from src.models.evaluate import optimize_threshold

        _, _, _, y_test = small_train_test
        y_test_bin = (y_test == 1).astype(int)
        rng = np.random.RandomState(42)
        y_proba = rng.rand(len(y_test_bin))
        result = optimize_threshold(y_test_bin, y_proba, pos_label=1)
        assert len(result) == 3

    def test_optimized_metrics_has_f1_fraud_key(self, small_train_test):
        from src.models.evaluate import optimize_threshold

        _, _, _, y_test = small_train_test
        y_test_bin = (y_test == 1).astype(int)
        rng = np.random.RandomState(42)
        y_proba = rng.rand(len(y_test_bin))
        _, _, metrics_opt = optimize_threshold(y_test_bin, y_proba, pos_label=1)
        assert "f1_fraud" in metrics_opt

    def test_optimized_pred_length_matches_input(self, small_train_test):
        from src.models.evaluate import optimize_threshold

        _, _, _, y_test = small_train_test
        y_test_bin = (y_test == 1).astype(int)
        rng = np.random.RandomState(42)
        y_proba = rng.rand(len(y_test_bin))
        _, y_pred_opt, _ = optimize_threshold(y_test_bin, y_proba, pos_label=1)
        assert len(y_pred_opt) == len(y_test_bin)


class TestOptimizeThresholdCostSensitive:
    """Tests for optimize_threshold_cost_sensitive."""

    @pytest.fixture
    def binary_test_data(self, small_train_test):
        """Binary y_test and random probabilities for cost-sensitive tests."""
        _, _, _, y_test = small_train_test
        y_test_bin = (y_test == 1).astype(int)
        rng = np.random.RandomState(99)
        y_proba = rng.rand(len(y_test_bin))
        return y_test_bin, y_proba

    def test_returns_four_items(self, binary_test_data):
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        result = optimize_threshold_cost_sensitive(y_true, y_proba)
        assert len(result) == 4

    def test_thresholds_in_unit_interval(self, binary_test_data):
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        f1_thr, cost_thr, _, _ = optimize_threshold_cost_sensitive(y_true, y_proba)
        assert 0.0 <= f1_thr <= 1.0
        assert 0.0 <= cost_thr <= 1.0

    def test_metrics_has_required_keys(self, binary_test_data):
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        _, _, _, metrics = optimize_threshold_cost_sensitive(y_true, y_proba)
        required = {"f1_fraud", "precision", "recall", "roc_auc", "net_value", "net_value_per_case"}
        assert required.issubset(set(metrics.keys()))

    def test_y_pred_cost_length_matches_input(self, binary_test_data):
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        _, _, y_pred_cost, _ = optimize_threshold_cost_sensitive(y_true, y_proba)
        assert len(y_pred_cost) == len(y_true)

    def test_y_pred_cost_is_binary(self, binary_test_data):
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        _, _, y_pred_cost, _ = optimize_threshold_cost_sensitive(y_true, y_proba)
        assert set(y_pred_cost).issubset({0, 1})

    def test_high_fraud_value_shifts_threshold_lower(self, binary_test_data):
        """High fraud_value → model should catch more fraud → lower threshold."""
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        _, thr_high, _, _ = optimize_threshold_cost_sensitive(
            y_true, y_proba, fraud_value=100_000, investigation_cost=50
        )
        _, thr_low, _, _ = optimize_threshold_cost_sensitive(
            y_true, y_proba, fraud_value=100, investigation_cost=50
        )
        # Higher fraud value should result in a lower or equal threshold
        assert thr_high <= thr_low + 0.01  # small tolerance for ties

    def test_negative_fraud_value_raises_value_error(self, binary_test_data):
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        with pytest.raises(ValueError, match="fraud_value"):
            optimize_threshold_cost_sensitive(y_true, y_proba, fraud_value=-1)

    def test_negative_investigation_cost_raises_value_error(self, binary_test_data):
        from src.models.evaluate import optimize_threshold_cost_sensitive

        y_true, y_proba = binary_test_data
        with pytest.raises(ValueError, match="investigation_cost"):
            optimize_threshold_cost_sensitive(y_true, y_proba, investigation_cost=-1)
