"""Unit tests for src/preprocessing/pipeline.py"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline


class TestCreatePreprocessingPipeline:
    def test_returns_sklearn_pipeline(self):
        from src.preprocessing.pipeline import create_preprocessing_pipeline

        pipe = create_preprocessing_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_required_steps(self):
        from src.preprocessing.pipeline import create_preprocessing_pipeline

        pipe = create_preprocessing_pipeline()
        assert "imputer" in pipe.named_steps
        assert "power_transform" in pipe.named_steps

    def test_fit_transform_preserves_shape(self, synthetic_features, feature_cols):
        from src.preprocessing.pipeline import create_preprocessing_pipeline

        pipe = create_preprocessing_pipeline()
        X = synthetic_features[feature_cols].values
        X_out = pipe.fit_transform(X)
        assert X_out.shape == X.shape

    def test_no_nan_after_transform(self, synthetic_features, feature_cols):
        from src.preprocessing.pipeline import create_preprocessing_pipeline

        # Inject NaN values to verify imputation
        X = synthetic_features[feature_cols].values.copy()
        X[0, 0] = np.nan
        X[1, 2] = np.nan
        pipe = create_preprocessing_pipeline()
        X_out = pipe.fit_transform(X)
        assert not np.isnan(X_out).any()


class TestPrepareFeaturesAndLabels:
    def test_excludes_metadata_columns(self, synthetic_labeled):
        from src.preprocessing.pipeline import prepare_features_and_labels

        X, y, time_steps, feature_cols = prepare_features_and_labels(synthetic_labeled)
        assert "txId" not in X.columns
        assert "class" not in X.columns
        assert "Time step" not in X.columns

    def test_returns_aligned_series(self, synthetic_labeled):
        from src.preprocessing.pipeline import prepare_features_and_labels

        X, y, time_steps, feature_cols = prepare_features_and_labels(synthetic_labeled)
        assert len(X) == len(y) == len(time_steps)

    def test_removes_infinite_values(self, synthetic_labeled):
        from src.preprocessing.pipeline import prepare_features_and_labels

        dirty = synthetic_labeled.copy()
        dirty.iloc[0, 2] = np.inf  # inject inf into a feature column
        X, y, time_steps, _ = prepare_features_and_labels(dirty)
        assert not np.isinf(X.values).any()

    def test_feature_cols_matches_x_columns(self, synthetic_labeled):
        from src.preprocessing.pipeline import prepare_features_and_labels

        X, _, _, feature_cols = prepare_features_and_labels(synthetic_labeled)
        assert list(X.columns) == feature_cols

    def test_custom_exclude_cols(self, synthetic_labeled):
        from src.preprocessing.pipeline import prepare_features_and_labels

        X, _, _, _ = prepare_features_and_labels(
            synthetic_labeled, exclude_cols=["txId", "class", "Time step"]
        )
        assert "txId" not in X.columns


class TestTemporalTrainTestSplit:
    def test_temporal_order_preserved(self, synthetic_labeled):
        from src.preprocessing.pipeline import (
            prepare_features_and_labels,
            temporal_train_test_split,
        )

        X, y, time_steps, _ = prepare_features_and_labels(synthetic_labeled)
        X_train, X_test, y_train, y_test, t_train, t_test = temporal_train_test_split(
            X, y, time_steps, split_ratio=0.8
        )
        # All training timesteps must be <= all test timesteps
        assert t_train.max() <= t_test.min()

    def test_split_ratio_approximately_correct(self, synthetic_labeled):
        from src.preprocessing.pipeline import (
            prepare_features_and_labels,
            temporal_train_test_split,
        )

        X, y, time_steps, _ = prepare_features_and_labels(synthetic_labeled)
        X_train, X_test, _, _, _, _ = temporal_train_test_split(
            X, y, time_steps, split_ratio=0.8
        )
        total = len(X_train) + len(X_test)
        assert abs(len(X_train) / total - 0.8) < 0.05

    def test_no_index_overlap_between_splits(self, synthetic_labeled):
        from src.preprocessing.pipeline import (
            prepare_features_and_labels,
            temporal_train_test_split,
        )

        X, y, time_steps, _ = prepare_features_and_labels(synthetic_labeled)
        X_train, X_test, _, _, _, _ = temporal_train_test_split(X, y, time_steps)
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_returns_six_objects(self, synthetic_labeled):
        from src.preprocessing.pipeline import (
            prepare_features_and_labels,
            temporal_train_test_split,
        )

        X, y, time_steps, _ = prepare_features_and_labels(synthetic_labeled)
        result = temporal_train_test_split(X, y, time_steps)
        assert len(result) == 6
