"""
Preprocessing utilities for feature engineering and data preparation.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

from src.utils.config import EXCLUDE_COLS, RANDOM_STATE, SPLIT_RATIO

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline() -> Pipeline:
    """Create sklearn preprocessing pipeline with median imputation and Yeo-Johnson transform.

    Returns:
        Unfitted sklearn Pipeline.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
        ]
    )


def temporal_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    time_steps: pd.Series,
    split_ratio: float = SPLIT_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split features and labels temporally (oldest rows go to train).

    Args:
        X: Feature DataFrame.
        y: Label Series.
        time_steps: Time step Series (must be aligned with X and y by index).
        split_ratio: Fraction of data to use for training (default: 0.8).

    Returns:
        X_train, X_test, y_train, y_test, time_train, time_test
    """
    sorted_indices = time_steps.sort_values().index
    X_sorted = X.loc[sorted_indices]
    y_sorted = y.loc[sorted_indices]
    time_steps_sorted = time_steps.loc[sorted_indices]

    split_idx = int(len(X_sorted) * split_ratio)

    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[split_idx:]
    time_train = time_steps_sorted.iloc[:split_idx]
    time_test = time_steps_sorted.iloc[split_idx:]

    logger.info(
        "Temporal split: %d train (timesteps %d–%d) / %d test (timesteps %d–%d)",
        len(X_train),
        int(time_train.min()),
        int(time_train.max()),
        len(X_test),
        int(time_test.min()),
        int(time_test.max()),
    )

    return X_train, X_test, y_train, y_test, time_train, time_test


def prepare_features_and_labels(
    labeled_data: pd.DataFrame,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """
    Extract features, labels, and time steps from a labeled transaction DataFrame.

    Removes metadata columns and rows containing infinite values.

    Args:
        labeled_data: DataFrame with labeled transactions (must contain 'txId',
            'class', and 'Time step' columns in addition to feature columns).
        exclude_cols: Columns to exclude from features. Defaults to
            ['txId', 'class', 'Time step'].

    Returns:
        X: Feature DataFrame.
        y: Label Series (values: 1=illicit, 2=licit).
        time_steps: Time step Series.
        feature_cols: List of feature column names used.
    """
    if exclude_cols is None:
        exclude_cols = EXCLUDE_COLS

    feature_cols = [col for col in labeled_data.columns if col not in exclude_cols]
    X = labeled_data[feature_cols].copy()
    y = labeled_data["class"].copy()
    time_steps = labeled_data["Time step"].copy()

    # Remove rows with infinite values — PowerTransformer cannot handle them
    inf_mask = np.isinf(X).any(axis=1)
    if inf_mask.sum() > 0:
        logger.warning("Removed %d rows containing infinite values", inf_mask.sum())
        X = X[~inf_mask]
        y = y[~inf_mask]
        time_steps = time_steps[~inf_mask]

    logger.info(
        "Feature matrix: %d samples × %d features", X.shape[0], X.shape[1]
    )

    return X, y, time_steps, feature_cols
