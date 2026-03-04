"""
Feature engineering and selection utilities for fraud detection.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold

from src.utils.config import ISOLATION_FOREST_PARAMS, RANDOM_STATE, STRUCTURAL_MISSING_COLS

logger = logging.getLogger(__name__)


def select_top_features_by_importance(
    importance_df: pd.DataFrame,
    n_features: int,
    importance_col: str = "shap_importance",
) -> list[str]:
    """Return the top n_features feature names ranked by importance_col."""
    selected = importance_df.nlargest(n_features, importance_col)["feature"].tolist()
    logger.info(
        "Selected top %d features by %s (range: %.4f – %.4f)",
        n_features,
        importance_col,
        importance_df[importance_col].nlargest(n_features).min(),
        importance_df[importance_col].nlargest(n_features).max(),
    )
    return selected


def remove_low_variance_features(
    X: pd.DataFrame,
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove near-constant features using a variance threshold.

    Args:
        X: Feature DataFrame.
        threshold: Minimum variance required to retain a feature (default: 0.01).

    Returns:
        X_reduced: DataFrame with low-variance features removed.
        removed_cols: List of column names that were dropped.
    """
    selector = VarianceThreshold(threshold=threshold)
    X_reduced_array = selector.fit_transform(X)
    kept_cols = [col for col, keep in zip(X.columns, selector.get_support()) if keep]
    removed_cols = [col for col in X.columns if col not in kept_cols]

    if removed_cols:
        logger.info(
            "Removed %d low-variance features (threshold=%.3f): %s…",
            len(removed_cols),
            threshold,
            removed_cols[:3],
        )

    return pd.DataFrame(X_reduced_array, columns=kept_cols, index=X.index), removed_cols


def run_feature_learning_curve(
    model_cls: type,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    importance_order: list[str],
    feature_counts: list[int] | None = None,
    random_state: int = RANDOM_STATE,
    **model_params: Any,
) -> pd.DataFrame:
    """
    Train a model with increasing feature subsets to find the performance–complexity
    sweet spot.

    Features are added in the order specified by importance_order (most important
    first).

    Args:
        model_cls: Model class to instantiate (e.g. XGBClassifier).
        X_train: Full training features array.
        y_train: Training labels.
        X_test: Full test features array.
        y_test: Test labels.
        feature_cols: Names of all features, aligned with X_train/X_test columns.
        importance_order: Feature names ordered from most to least important.
        feature_counts: List of feature subset sizes to evaluate. Defaults to
            [5, 8, 12, 20, 30, 50, 75, 100, 150, len(feature_cols)].
        random_state: Random seed.
        **model_params: Additional keyword arguments forwarded to model_cls.

    Returns:
        DataFrame with columns: n_features, fraud_f1_default, fraud_f1_optimized,
        fraud_precision, fraud_recall, roc_auc.
    """
    from src.models.evaluate import evaluate_classifier, optimize_threshold

    if feature_counts is None:
        max_n = len(feature_cols)
        feature_counts = [n for n in [5, 8, 12, 20, 30, 50, 75, 100, 150, max_n] if n <= max_n]

    col_to_idx = {col: i for i, col in enumerate(feature_cols)}
    results = []

    for n in feature_counts:
        top_n = importance_order[:n]
        indices = [col_to_idx[c] for c in top_n if c in col_to_idx]

        X_train_sub = X_train[:, indices]
        X_test_sub = X_test[:, indices]

        model = model_cls(random_state=random_state, **model_params)
        model.fit(X_train_sub, y_train)

        y_proba = model.predict_proba(X_test_sub)[:, 1]
        y_pred = model.predict(X_test_sub)

        metrics_default = evaluate_classifier(y_test, y_pred, y_proba, pos_label=1)
        _, _, metrics_opt = optimize_threshold(y_test, y_proba, pos_label=1)

        results.append(
            {
                "n_features": n,
                "fraud_f1_default": metrics_default["f1_fraud"],
                "fraud_f1_optimized": metrics_opt["f1_fraud"],
                "fraud_precision": metrics_opt["precision"],
                "fraud_recall": metrics_opt["recall"],
                "roc_auc": metrics_default["roc_auc"],
            }
        )
        logger.info(
            "n_features=%d → F1=%.4f (opt=%.4f), AUC=%.4f",
            n,
            metrics_default["f1_fraud"],
            metrics_opt["f1_fraud"],
            metrics_default["roc_auc"],
        )

    return pd.DataFrame(results)


def analyze_missingness_vs_labels(
    labeled_df: pd.DataFrame,
    missing_cols: list[str],
) -> dict[str, float | int]:
    """
    Confirm the missingness–fraud correlation statistically.

    Args:
        labeled_df: Merged labeled DataFrame with 'class' column (1=illicit, 2=licit).
        missing_cols: Structural columns expected to be NaN only for licit transactions.

    Returns:
        Dict with counts, percentages, point-biserial r, p_biserial, odds_ratio,
        p_fisher. All values are zero if none of missing_cols exist in labeled_df.
    """
    cols_present = [c for c in missing_cols if c in labeled_df.columns]

    zero_result: dict[str, float | int] = {
        "n_illicit_with_missing": 0,
        "n_licit_with_missing": 0,
        "pct_illicit_missing": 0.0,
        "pct_licit_missing": 0.0,
        "r_pointbiserial": 0.0,
        "p_biserial": 1.0,
        "odds_ratio": 0.0,
        "p_fisher": 1.0,
    }

    if not cols_present:
        logger.warning(
            "analyze_missingness_vs_labels: none of the %d requested columns are present "
            "in labeled_df — returning zero result.",
            len(missing_cols),
        )
        return zero_result

    has_missing = labeled_df[cols_present].isna().any(axis=1)
    is_illicit = (labeled_df["class"] == 1).astype(int)

    n_illicit = int(is_illicit.sum())
    n_licit = int((is_illicit == 0).sum())

    n_illicit_with_missing = int((has_missing & (is_illicit == 1)).sum())
    n_licit_with_missing = int((has_missing & (is_illicit == 0)).sum())

    pct_illicit_missing = n_illicit_with_missing / max(n_illicit, 1) * 100
    pct_licit_missing = n_licit_with_missing / max(n_licit, 1) * 100

    r, p_biserial = stats.pointbiserialr(has_missing.astype(float), is_illicit.astype(float))

    # Fisher exact: [[illicit+missing, illicit+complete], [licit+missing, licit+complete]]
    a = n_illicit_with_missing
    b = n_illicit - n_illicit_with_missing
    c = n_licit_with_missing
    d = n_licit - n_licit_with_missing
    odds_ratio, p_fisher = stats.fisher_exact([[a, b], [c, d]])

    result: dict[str, float | int] = {
        "n_illicit_with_missing": n_illicit_with_missing,
        "n_licit_with_missing": n_licit_with_missing,
        "pct_illicit_missing": round(pct_illicit_missing, 4),
        "pct_licit_missing": round(pct_licit_missing, 4),
        "r_pointbiserial": round(float(r), 6),
        "p_biserial": round(float(p_biserial), 8),
        "odds_ratio": round(float(odds_ratio), 6),
        "p_fisher": round(float(p_fisher), 8),
    }

    logger.info(
        "Missingness analysis: illicit_missing=%d (%.2f%%), licit_missing=%d (%.2f%%)",
        n_illicit_with_missing,
        pct_illicit_missing,
        n_licit_with_missing,
        pct_licit_missing,
    )
    return result


def add_missingness_indicator(
    df: pd.DataFrame,
    missing_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add a binary `is_structural_missing` column (1 = any structural col is NaN).

    Args:
        df: DataFrame to augment (typically the full labeled DataFrame).
        missing_cols: Structural columns to check for NaN. Defaults to
            STRUCTURAL_MISSING_COLS from config.

    Returns:
        Copy of df with an additional int8 column `is_structural_missing`.
    """
    if missing_cols is None:
        missing_cols = STRUCTURAL_MISSING_COLS

    result = df.copy()
    cols_to_check = [c for c in missing_cols if c in result.columns]

    if not cols_to_check:
        logger.warning(
            "add_missingness_indicator: none of the %d structural columns found; "
            "setting is_structural_missing=0 for all rows.",
            len(missing_cols),
        )
        result["is_structural_missing"] = np.int8(0)
        return result

    result["is_structural_missing"] = result[cols_to_check].isna().any(axis=1).astype(np.int8)
    n_flagged = int(result["is_structural_missing"].sum())
    logger.info("Flagged %d / %d rows as structurally missing", n_flagged, len(result))
    return result


def compute_graph_features(
    df: pd.DataFrame,
    edgelist_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add in-degree, out-degree, and total degree features from the transaction edgelist.

    Args:
        df: Labeled transaction DataFrame (must contain 'txId').
        edgelist_df: Directed edge list with columns 'txId1' (source) and 'txId2' (dest).

    Returns:
        Copy of df with three new int32 columns:
            graph_in_degree, graph_out_degree, graph_total_degree.

    Raises:
        ValueError: If edgelist_df is missing 'txId1' or 'txId2' columns.
    """
    required_cols = {"txId1", "txId2"}
    if not required_cols.issubset(edgelist_df.columns):
        missing = required_cols - set(edgelist_df.columns)
        raise ValueError(f"edgelist_df is missing required columns: {missing}")

    in_deg = edgelist_df.groupby("txId2").size().rename("graph_in_degree")
    out_deg = edgelist_df.groupby("txId1").size().rename("graph_out_degree")

    result = df.copy()
    result = result.join(in_deg, on="txId")
    result = result.join(out_deg, on="txId")

    result["graph_in_degree"] = result["graph_in_degree"].fillna(0).astype(np.int32)
    result["graph_out_degree"] = result["graph_out_degree"].fillna(0).astype(np.int32)
    result["graph_total_degree"] = result["graph_in_degree"] + result["graph_out_degree"]

    n_isolated = int((result["graph_total_degree"] == 0).sum())
    logger.info("Graph degrees computed; %d isolated nodes", n_isolated)
    return result


def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float | None = None,
    random_state: int | None = None,
) -> IsolationForest:
    """Fit an IsolationForest on X_train and return it."""
    params = ISOLATION_FOREST_PARAMS.copy()
    if contamination is not None:
        params["contamination"] = contamination
    if random_state is not None:
        params["random_state"] = random_state

    clf = IsolationForest(**params)
    clf.fit(X_train)

    logger.info("IsolationForest fitted on %s", X_train.shape)
    return clf


def add_anomaly_score_feature(
    X: np.ndarray,
    isolation_forest_model: IsolationForest,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    # decision_function gives continuous score (higher = more normal), not binary predict
    scores = isolation_forest_model.decision_function(X).reshape(-1, 1)
    X_with_score = np.hstack([X, scores])
    updated_feature_cols = list(feature_cols) + ["anomaly_score"]

    logger.info(
        "add_anomaly_score_feature: scores mean=%.4f, std=%.4f",
        float(scores.mean()),
        float(scores.std()),
    )
    return X_with_score, updated_feature_cols


def add_temporal_features(
    df: pd.DataFrame,
    labeled_train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Adds ``timestep`` (raw integer, 1–49) and ``rolling_fraud_rate_3step``
    (3-step lookback fraud rate) to df.

    The rolling rate is computed from labeled_train only to avoid leakage —
    for each timestep t, we look at t-3 through t-1 in the training window.
    Pass labeled_train=None during inference; the rate column will be NaN.
    """
    # Note: uses training data only to avoid leakage — test rows are mapped from
    # a lookup table built on labeled_train
    if "Time step" not in df.columns:
        raise KeyError("df must contain a 'Time step' column")

    result = df.copy()
    result["timestep"] = result["Time step"].astype(np.float64)

    if labeled_train is None or len(labeled_train) == 0:
        result["rolling_fraud_rate_3step"] = np.nan
        logger.info(
            "add_temporal_features: labeled_train not provided; "
            "rolling_fraud_rate_3step set to NaN for all %d rows.",
            len(result),
        )
        return result

    # Compute per-timestep fraud rate from training data (exclude unknown class 3)
    from src.utils.config import ILLICIT_LABEL, UNKNOWN_LABEL

    train_labeled = labeled_train[labeled_train["class"] != UNKNOWN_LABEL].copy()
    train_labeled["is_illicit"] = (train_labeled["class"] == ILLICIT_LABEL).astype(float)
    step_fraud_rate = (
        train_labeled.groupby("Time step")["is_illicit"].mean().sort_index()
    )
    overall_rate = float(train_labeled["is_illicit"].mean())

    # Build rolling 3-step lookup: rate for step t = mean of steps t-3, t-2, t-1
    all_steps = sorted(step_fraud_rate.index.tolist())
    rolling_lookup: dict[int, float] = {}
    for step in all_steps:
        prior_steps = [s for s in all_steps if step - 3 <= s < step]
        if prior_steps:
            rolling_lookup[step] = float(step_fraud_rate.loc[prior_steps].mean())
        else:
            rolling_lookup[step] = overall_rate

    result["rolling_fraud_rate_3step"] = (
        result["Time step"].map(rolling_lookup).fillna(overall_rate)
    )

    logger.info(
        "add_temporal_features: timestep range [%d, %d], "
        "rolling_fraud_rate_3step range [%.4f, %.4f] (overall_rate=%.4f)",
        int(result["timestep"].min()),
        int(result["timestep"].max()),
        float(result["rolling_fraud_rate_3step"].min()),
        float(result["rolling_fraud_rate_3step"].max()),
        overall_rate,
    )
    return result


def pseudo_label_unlabeled(
    model: Any,
    X_unlabeled: np.ndarray,
    thresh_fraud: float = 0.97,  # only label illicit if p >= 0.97 (very tight — avoid FP pseudo-labels)
    thresh_licit: float = 0.03,  # only label licit if p <= 0.03 (symmetric bound)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score unlabeled transactions and return high-confidence pseudo-labels for self-training.
    Only predictions beyond tight confidence bounds are kept.
    Returns (X_pseudo, y_pseudo) where y_pseudo is binary (1=illicit, 0=licit).
    """
    if thresh_licit >= thresh_fraud:
        raise ValueError(
            f"thresh_licit ({thresh_licit}) must be strictly less than "
            f"thresh_fraud ({thresh_fraud})"
        )

    if len(X_unlabeled) == 0:
        logger.info("pseudo_label_unlabeled: empty input array — returning empty arrays.")
        return X_unlabeled, np.array([], dtype=np.int32)

    y_proba = model.predict_proba(X_unlabeled)[:, 1]

    illicit_mask = y_proba >= thresh_fraud
    licit_mask = y_proba <= thresh_licit
    keep_mask = illicit_mask | licit_mask

    X_pseudo = X_unlabeled[keep_mask]
    y_pseudo = np.where(illicit_mask[keep_mask], 1, 0).astype(np.int32)

    n_illicit_pseudo = int(illicit_mask.sum())
    n_licit_pseudo = int(licit_mask.sum())
    logger.info(
        "Pseudo-labels: %d illicit, %d licit of %d unlabeled",
        n_illicit_pseudo,
        n_licit_pseudo,
        len(X_unlabeled),
    )
    return X_pseudo, y_pseudo


def compute_class_imbalance_ratio(y: pd.Series) -> float:
    counts = y.value_counts()
    ratio = float(counts.max() / counts.min())
    logger.info(
        "Class imbalance ratio: %.2f:1 (majority=%d, minority=%d)",
        ratio,
        int(counts.max()),
        int(counts.min()),
    )
    return ratio
