"""Unit tests for src/features/feature_engineering.py — new feature engineering functions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest, RandomForestClassifier


class TestAnalyzeMissingnessVsLabels:
    """Tests for analyze_missingness_vs_labels."""

    REQUIRED_KEYS = {
        "n_illicit_with_missing",
        "n_licit_with_missing",
        "pct_illicit_missing",
        "pct_licit_missing",
        "r_pointbiserial",
        "p_biserial",
        "odds_ratio",
        "p_fisher",
    }

    def test_returns_dict_with_required_keys(self, synthetic_labeled):
        from src.features.feature_engineering import analyze_missingness_vs_labels

        result = analyze_missingness_vs_labels(synthetic_labeled, ["feature_1"])
        assert set(result.keys()) == self.REQUIRED_KEYS

    def test_no_structural_cols_returns_zero_dict(self, synthetic_labeled):
        """Guard case: none of the requested columns exist → zero result, no error."""
        from src.features.feature_engineering import analyze_missingness_vs_labels

        result = analyze_missingness_vs_labels(synthetic_labeled, ["nonexistent_col_xyz"])
        assert result["n_illicit_with_missing"] == 0
        assert result["n_licit_with_missing"] == 0
        assert result["pct_illicit_missing"] == 0.0

    def test_artificial_nan_data_detects_licit_missing(self):
        """With synthetic NaN data in licit rows: licit_missing > 0."""
        from src.features.feature_engineering import analyze_missingness_vs_labels

        rng = np.random.RandomState(0)
        n = 100
        df = pd.DataFrame(
            {
                "txId": np.arange(n),
                "class": rng.choice([1, 2], size=n, p=[0.1, 0.9]),
                "val": rng.randn(n),
            }
        )
        # Make val NaN only for licit rows
        df.loc[df["class"] == 2, "val"] = np.nan

        result = analyze_missingness_vs_labels(df, ["val"])
        assert result["n_licit_with_missing"] > 0
        assert result["pct_licit_missing"] > 0.0

    def test_does_not_mutate_dataframe(self, synthetic_labeled):
        from src.features.feature_engineering import analyze_missingness_vs_labels

        original_cols = list(synthetic_labeled.columns)
        analyze_missingness_vs_labels(synthetic_labeled, ["feature_1"])
        assert list(synthetic_labeled.columns) == original_cols


class TestAddMissingnessIndicator:
    """Tests for add_missingness_indicator."""

    def test_output_has_is_structural_missing_column(self, synthetic_labeled):
        from src.features.feature_engineering import add_missingness_indicator

        result = add_missingness_indicator(synthetic_labeled)
        assert "is_structural_missing" in result.columns

    def test_output_shape_is_original_plus_one(self, synthetic_labeled):
        from src.features.feature_engineering import add_missingness_indicator

        result = add_missingness_indicator(synthetic_labeled)
        assert result.shape == (synthetic_labeled.shape[0], synthetic_labeled.shape[1] + 1)

    def test_values_are_binary(self, synthetic_labeled):
        from src.features.feature_engineering import add_missingness_indicator

        result = add_missingness_indicator(synthetic_labeled)
        assert set(result["is_structural_missing"].unique()).issubset({0, 1})

    def test_no_structural_cols_returns_all_zeros(self, synthetic_labeled):
        """Synthetic data has no structural cols → all zeros, no error."""
        from src.features.feature_engineering import add_missingness_indicator

        result = add_missingness_indicator(synthetic_labeled, missing_cols=["nonexistent_col"])
        assert result["is_structural_missing"].sum() == 0

    def test_does_not_mutate_original_dataframe(self, synthetic_labeled):
        from src.features.feature_engineering import add_missingness_indicator

        original_shape = synthetic_labeled.shape
        add_missingness_indicator(synthetic_labeled)
        assert synthetic_labeled.shape == original_shape
        assert "is_structural_missing" not in synthetic_labeled.columns

    def test_detects_nan_correctly(self):
        """With explicit NaN in a specified column, flag should be 1."""
        from src.features.feature_engineering import add_missingness_indicator

        df = pd.DataFrame(
            {
                "txId": [1, 2, 3],
                "class": [1, 2, 2],
                "val": [1.0, np.nan, 3.0],
            }
        )
        result = add_missingness_indicator(df, missing_cols=["val"])
        assert result.loc[result["txId"] == 2, "is_structural_missing"].values[0] == 1
        assert result.loc[result["txId"] == 1, "is_structural_missing"].values[0] == 0


class TestComputeGraphFeatures:
    """Tests for compute_graph_features."""

    def test_output_has_degree_columns(self, synthetic_labeled, synthetic_edgelist):
        from src.features.feature_engineering import compute_graph_features

        result = compute_graph_features(synthetic_labeled, synthetic_edgelist)
        assert "graph_in_degree" in result.columns
        assert "graph_out_degree" in result.columns
        assert "graph_total_degree" in result.columns

    def test_degree_values_are_non_negative(self, synthetic_labeled, synthetic_edgelist):
        from src.features.feature_engineering import compute_graph_features

        result = compute_graph_features(synthetic_labeled, synthetic_edgelist)
        assert (result["graph_in_degree"] >= 0).all()
        assert (result["graph_out_degree"] >= 0).all()
        assert (result["graph_total_degree"] >= 0).all()

    def test_isolated_nodes_get_degree_zero(self, synthetic_labeled):
        """Transactions not in the edgelist should have degree 0."""
        from src.features.feature_engineering import compute_graph_features

        # Edgelist uses txIds that don't exist in synthetic_labeled (txIds 1-200)
        edgelist = pd.DataFrame({"txId1": [99999], "txId2": [99998]})
        result = compute_graph_features(synthetic_labeled, edgelist)
        assert (result["graph_in_degree"] == 0).all()
        assert (result["graph_out_degree"] == 0).all()

    def test_total_degree_is_sum_of_in_and_out(self, synthetic_labeled, synthetic_edgelist):
        from src.features.feature_engineering import compute_graph_features

        result = compute_graph_features(synthetic_labeled, synthetic_edgelist)
        expected = result["graph_in_degree"] + result["graph_out_degree"]
        pd.testing.assert_series_equal(result["graph_total_degree"], expected, check_names=False)

    def test_raises_value_error_for_missing_columns(self, synthetic_labeled):
        from src.features.feature_engineering import compute_graph_features

        bad_edgelist = pd.DataFrame({"source": [1], "target": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            compute_graph_features(synthetic_labeled, bad_edgelist)

    def test_does_not_mutate_original_dataframe(self, synthetic_labeled, synthetic_edgelist):
        from src.features.feature_engineering import compute_graph_features

        original_cols = list(synthetic_labeled.columns)
        compute_graph_features(synthetic_labeled, synthetic_edgelist)
        assert list(synthetic_labeled.columns) == original_cols


class TestIsolationForest:
    """Tests for fit_isolation_forest and add_anomaly_score_feature."""

    @pytest.fixture
    def small_X_train(self):
        rng = np.random.RandomState(42)
        return rng.randn(80, 10).astype(np.float32)

    @pytest.fixture
    def small_X_test(self):
        rng = np.random.RandomState(99)
        return rng.randn(20, 10).astype(np.float32)

    @pytest.fixture
    def fitted_iso_forest(self, small_X_train):
        from src.features.feature_engineering import fit_isolation_forest

        return fit_isolation_forest(small_X_train)

    def test_fit_isolation_forest_returns_fitted_model(self, small_X_train):
        from src.features.feature_engineering import fit_isolation_forest

        model = fit_isolation_forest(small_X_train)
        assert isinstance(model, IsolationForest)
        # A fitted model has estimators_
        assert hasattr(model, "estimators_")

    def test_add_anomaly_score_adds_one_column(self, small_X_train, fitted_iso_forest):
        from src.features.feature_engineering import add_anomaly_score_feature

        feature_cols = [f"f{i}" for i in range(small_X_train.shape[1])]
        X_out, cols_out = add_anomaly_score_feature(small_X_train, fitted_iso_forest, feature_cols)
        assert X_out.shape == (small_X_train.shape[0], small_X_train.shape[1] + 1)
        assert len(cols_out) == len(feature_cols) + 1

    def test_anomaly_score_col_name_is_anomaly_score(self, small_X_train, fitted_iso_forest):
        from src.features.feature_engineering import add_anomaly_score_feature

        feature_cols = [f"f{i}" for i in range(small_X_train.shape[1])]
        _, cols_out = add_anomaly_score_feature(small_X_train, fitted_iso_forest, feature_cols)
        assert cols_out[-1] == "anomaly_score"

    def test_anomaly_scores_are_finite_floats(self, small_X_train, fitted_iso_forest):
        from src.features.feature_engineering import add_anomaly_score_feature

        feature_cols = [f"f{i}" for i in range(small_X_train.shape[1])]
        X_out, _ = add_anomaly_score_feature(small_X_train, fitted_iso_forest, feature_cols)
        scores = X_out[:, -1]
        assert np.isfinite(scores).all()
        assert scores.dtype in [np.float32, np.float64]

    def test_test_scores_use_train_fitted_model(self, small_X_train, small_X_test, fitted_iso_forest):
        """Verify test scores come from the train-fitted model, not a refit."""
        from src.features.feature_engineering import add_anomaly_score_feature

        feature_cols = [f"f{i}" for i in range(small_X_train.shape[1])]
        X_test_out, _ = add_anomaly_score_feature(small_X_test, fitted_iso_forest, feature_cols)
        # Scores should be reproducible for the same inputs and fitted model
        X_test_out2, _ = add_anomaly_score_feature(small_X_test, fitted_iso_forest, feature_cols)
        np.testing.assert_array_equal(X_test_out[:, -1], X_test_out2[:, -1])

    def test_contamination_override(self, small_X_train):
        from src.features.feature_engineering import fit_isolation_forest

        model = fit_isolation_forest(small_X_train, contamination=0.05)
        assert model.contamination == 0.05

    def test_random_state_override(self, small_X_train):
        from src.features.feature_engineering import fit_isolation_forest

        model = fit_isolation_forest(small_X_train, random_state=7)
        assert model.random_state == 7


class TestAddTemporalFeatures:
    """Tests for add_temporal_features."""

    def test_adds_timestep_column(self, synthetic_labeled):
        from src.features.feature_engineering import add_temporal_features

        result = add_temporal_features(synthetic_labeled)
        assert "timestep" in result.columns

    def test_adds_rolling_fraud_rate_column(self, synthetic_labeled):
        from src.features.feature_engineering import add_temporal_features

        result = add_temporal_features(synthetic_labeled, labeled_train=synthetic_labeled)
        assert "rolling_fraud_rate_3step" in result.columns

    def test_timestep_values_match_time_step(self, synthetic_labeled):
        from src.features.feature_engineering import add_temporal_features

        result = add_temporal_features(synthetic_labeled)
        import numpy as np
        np.testing.assert_array_equal(
            result["timestep"].values, synthetic_labeled["Time step"].values.astype(float)
        )

    def test_rolling_fraud_rate_in_unit_interval(self, synthetic_labeled):
        from src.features.feature_engineering import add_temporal_features

        result = add_temporal_features(synthetic_labeled, labeled_train=synthetic_labeled)
        rates = result["rolling_fraud_rate_3step"]
        assert (rates >= 0.0).all() and (rates <= 1.0).all()

    def test_no_labeled_train_gives_nan_rolling(self, synthetic_labeled):
        from src.features.feature_engineering import add_temporal_features

        result = add_temporal_features(synthetic_labeled, labeled_train=None)
        assert result["rolling_fraud_rate_3step"].isna().all()

    def test_raises_key_error_for_missing_time_step(self):
        from src.features.feature_engineering import add_temporal_features

        df = pd.DataFrame({"txId": [1, 2], "feature_1": [0.1, 0.2]})
        with pytest.raises(KeyError, match="Time step"):
            add_temporal_features(df)

    def test_does_not_mutate_original_dataframe(self, synthetic_labeled):
        from src.features.feature_engineering import add_temporal_features

        original_cols = list(synthetic_labeled.columns)
        add_temporal_features(synthetic_labeled, labeled_train=synthetic_labeled)
        assert list(synthetic_labeled.columns) == original_cols

    def test_output_shape_is_original_plus_two(self, synthetic_labeled):
        from src.features.feature_engineering import add_temporal_features

        result = add_temporal_features(synthetic_labeled, labeled_train=synthetic_labeled)
        assert result.shape == (synthetic_labeled.shape[0], synthetic_labeled.shape[1] + 2)

    def test_rolling_rate_uses_only_prior_steps(self, synthetic_labeled):
        """For the first timestep, there are no prior steps → falls back to overall rate."""
        from src.features.feature_engineering import add_temporal_features

        result = add_temporal_features(synthetic_labeled, labeled_train=synthetic_labeled)
        # First timestep row should have the overall training fraud rate
        first_step = synthetic_labeled["Time step"].min()
        first_step_rate = result.loc[
            result["Time step"] == first_step, "rolling_fraud_rate_3step"
        ].iloc[0]
        # Overall rate from the training data
        overall_rate = (synthetic_labeled["class"] == 1).mean()
        assert abs(first_step_rate - overall_rate) < 1e-6


class TestPseudoLabelUnlabeled:
    """Tests for pseudo_label_unlabeled."""

    @pytest.fixture
    def fitted_model(self):
        """Small RF fitted on binary-labelled synthetic data."""
        rng = np.random.RandomState(0)
        X = rng.randn(100, 5).astype(np.float32)
        y = rng.choice([0, 1], size=100, p=[0.9, 0.1])
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def X_unlabeled(self):
        rng = np.random.RandomState(7)
        return rng.randn(50, 5).astype(np.float32)

    def test_returns_two_arrays(self, fitted_model, X_unlabeled):
        from src.features.feature_engineering import pseudo_label_unlabeled

        result = pseudo_label_unlabeled(fitted_model, X_unlabeled)
        assert len(result) == 2

    def test_X_pseudo_and_y_pseudo_same_length(self, fitted_model, X_unlabeled):
        from src.features.feature_engineering import pseudo_label_unlabeled

        X_pseudo, y_pseudo = pseudo_label_unlabeled(fitted_model, X_unlabeled)
        assert len(X_pseudo) == len(y_pseudo)

    def test_pseudo_labels_are_binary(self, fitted_model, X_unlabeled):
        from src.features.feature_engineering import pseudo_label_unlabeled

        _, y_pseudo = pseudo_label_unlabeled(
            fitted_model, X_unlabeled, thresh_fraud=0.5, thresh_licit=0.49
        )
        assert set(y_pseudo).issubset({0, 1})

    def test_tight_thresholds_filter_out_uncertain_samples(self, fitted_model, X_unlabeled):
        """At very tight thresholds, fewer (or zero) pseudo-labels should be kept."""
        from src.features.feature_engineering import pseudo_label_unlabeled

        X_tight, _ = pseudo_label_unlabeled(
            fitted_model, X_unlabeled, thresh_fraud=0.9999, thresh_licit=0.0001
        )
        X_loose, _ = pseudo_label_unlabeled(
            fitted_model, X_unlabeled, thresh_fraud=0.5, thresh_licit=0.49
        )
        assert len(X_tight) <= len(X_loose)

    def test_invalid_thresholds_raise_value_error(self, fitted_model, X_unlabeled):
        from src.features.feature_engineering import pseudo_label_unlabeled

        with pytest.raises(ValueError, match="thresh_licit"):
            pseudo_label_unlabeled(
                fitted_model, X_unlabeled, thresh_fraud=0.3, thresh_licit=0.5
            )

    def test_X_pseudo_has_same_n_columns_as_input(self, fitted_model, X_unlabeled):
        from src.features.feature_engineering import pseudo_label_unlabeled

        X_pseudo, _ = pseudo_label_unlabeled(
            fitted_model, X_unlabeled, thresh_fraud=0.5, thresh_licit=0.49
        )
        assert X_pseudo.shape[1] == X_unlabeled.shape[1]

    def test_empty_unlabeled_returns_empty_arrays(self, fitted_model):
        from src.features.feature_engineering import pseudo_label_unlabeled

        X_empty = np.zeros((0, 5), dtype=np.float32)
        X_pseudo, y_pseudo = pseudo_label_unlabeled(fitted_model, X_empty)
        assert len(X_pseudo) == 0
        assert len(y_pseudo) == 0
