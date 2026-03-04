"""
Model training utilities for fraud detection.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import shap

from src.utils.config import RANDOM_STATE, SHAP_SAMPLE_SIZE

logger = logging.getLogger(__name__)


def train_classifier(
    model: Any,
    X_train: NDArray,
    y_train: NDArray | pd.Series,
    X_test: NDArray,
    y_test: NDArray | pd.Series,
    model_name: str,
) -> tuple[Any, NDArray, NDArray]:
    """
    Train a classifier and generate predictions on the test set.

    Args:
        model: Unfitted ML model instance (e.g. RandomForestClassifier, XGBClassifier).
        X_train: Training features array.
        y_train: Training labels.
        X_test: Test features array.
        y_test: Test labels (unused during training; kept for API consistency).
        model_name: Display name used for logging.

    Returns:
        model: Fitted model.
        y_pred: Hard predictions on the test set.
        y_proba: Probability scores for the positive class on the test set.
    """
    logger.info(
        "Training %s on %d samples (%d features)...", model_name, len(X_train), X_train.shape[1]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        proba_output = model.predict_proba(X_test)
        if len(proba_output.shape) == 2 and proba_output.shape[1] == 2:
            y_proba = proba_output[:, 1]
        else:
            y_proba = proba_output
    else:
        y_proba = y_pred.astype(float)

    logger.info("Training complete. Predicting on %d test samples.", len(X_test))
    return model, y_pred, y_proba


def train_random_forest_with_gini(
    X_train: NDArray,
    y_train: NDArray | pd.Series,
    X_test: NDArray,
    y_test: NDArray | pd.Series,
    feature_cols: list[str],
    **rf_params: Any,
) -> tuple[RandomForestClassifier, NDArray, NDArray, pd.DataFrame]:
    """
    Train a Random Forest and compute Gini feature importance.
    Returns (rf, y_pred, y_proba, gini_importance_df).
    """
    logger.info("Training Random Forest (n_estimators=%s)...", rf_params.get("n_estimators", "?"))
    rf = RandomForestClassifier(**rf_params)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 0]

    gini_importance = pd.DataFrame(
        {"feature": feature_cols, "gini_importance": rf.feature_importances_}
    ).sort_values("gini_importance", ascending=False)

    logger.info(
        "Top Gini feature: %s (%.4f)",
        gini_importance.iloc[0]["feature"],
        gini_importance.iloc[0]["gini_importance"],
    )

    return rf, y_pred, y_proba, gini_importance


def compute_permutation_importance(
    model: Any,
    X_test: NDArray,
    y_test: NDArray | pd.Series,
    feature_cols: list[str],
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Compute permutation importance for a trained model on the test set.

    Args:
        model: Fitted model with a predict method.
        X_test: Test features array.
        y_test: Test labels.
        feature_cols: Feature names aligned with X_test columns.
        n_repeats: Number of permutation repetitions per feature.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns ['feature', 'perm_importance', 'perm_std'],
        sorted by perm_importance descending.
    """
    logger.info("Computing permutation importance (%d repeats)...", n_repeats)
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    perm_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "perm_importance": result.importances_mean,
            "perm_std": result.importances_std,
        }
    ).sort_values("perm_importance", ascending=False)

    return perm_df


def compute_shap_values(
    model: Any,
    X_test: NDArray,
    feature_cols: list[str],
    sample_size: int = SHAP_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[NDArray, pd.DataFrame, NDArray]:
    """
    Compute SHAP values using a TreeExplainer for model interpretability.

    Args:
        model: Fitted tree-based model (RandomForest or XGBoost).
        X_test: Test features array.
        feature_cols: Feature names aligned with X_test columns.
        sample_size: Maximum number of test samples to use for SHAP computation.
        random_state: Random seed for reproducible subsampling.

    Returns:
        shap_values_fraud: SHAP values for the fraud class, shape (n_samples, n_features).
        shap_importance: DataFrame with columns ['feature', 'shap_importance'],
            sorted descending.
        X_test_sample: Subsampled test array used for SHAP computation.
    """
    explainer = shap.TreeExplainer(model)

    if len(X_test) > sample_size:
        rng = np.random.RandomState(random_state)
        sample_indices = rng.choice(len(X_test), size=sample_size, replace=False)
        X_test_sample = X_test[sample_indices]
        logger.info("SHAP computation on %d samples (subsampled from %d)", sample_size, len(X_test))
    else:
        X_test_sample = X_test
        logger.info("SHAP computation on all %d test samples", len(X_test))

    shap_values = explainer.shap_values(X_test_sample)

    # Extract fraud class SHAP values
    if len(shap_values.shape) == 3:
        shap_values_fraud = shap_values[:, :, 0]
    else:
        shap_values_fraud = shap_values

    mean_abs_shap = np.abs(shap_values_fraud).mean(axis=0)
    shap_importance = pd.DataFrame(
        {"feature": feature_cols, "shap_importance": mean_abs_shap}
    ).sort_values("shap_importance", ascending=False)

    logger.info(
        "Top SHAP feature: %s (mean |SHAP|=%.4f)",
        shap_importance.iloc[0]["feature"],
        shap_importance.iloc[0]["shap_importance"],
    )

    return shap_values_fraud, shap_importance, X_test_sample
