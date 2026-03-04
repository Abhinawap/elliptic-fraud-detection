"""
Inference utilities for loading trained models and generating predictions.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def load_model(model_path: Path) -> Any:
    """
    Load a serialized model from disk using joblib.

    Args:
        model_path: Path to the saved model file (.joblib or .pkl).

    Returns:
        Deserialized model object.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    import joblib

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    logger.info("Loaded model from %s", model_path)
    return model


def predict_fraud_probability(
    model: Any,
    X: NDArray | pd.DataFrame,
) -> NDArray:
    """
    Return fraud probability scores for each transaction.

    Args:
        model: Fitted classification model with a predict_proba method.
        X: Feature array or DataFrame.

    Returns:
        1D array of fraud probability scores in [0, 1].
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
    logger.warning("Model lacks predict_proba; falling back to predict()")
    return model.predict(X).astype(float)


def predict_with_threshold(
    model: Any,
    X: NDArray | pd.DataFrame,
    threshold: float = 0.5,  # default 0.5 is rarely right — use optimize_threshold output
) -> tuple[NDArray, NDArray]:
    """Returns (binary predictions, fraud probabilities) at the given threshold."""
    probabilities = predict_fraud_probability(model, X)
    predictions = (probabilities >= threshold).astype(int)

    fraud_count = int(predictions.sum())
    total = len(predictions)
    logger.info(
        "Predicted %d fraud / %d legit out of %d transactions (threshold=%.3f)",
        fraud_count,
        total - fraud_count,
        total,
        threshold,
    )
    return predictions, probabilities


def batch_predict(
    model: Any,
    data_path: Path,
    pipeline: Any,
    feature_cols: list[str],
    threshold: float = 0.5,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Load raw transaction data, preprocess with a fitted pipeline, and predict fraud.

    Args:
        model: Fitted classification model.
        data_path: Path to a CSV file with raw transaction features.
        pipeline: Fitted sklearn preprocessing pipeline (imputer + transformer).
        feature_cols: Feature columns to use (must be present in data_path).
        threshold: Fraud probability cutoff.
        output_path: If provided, saves prediction results to this CSV path.

    Returns:
        DataFrame with columns: txId (if present), fraud_probability, is_fraud.
    """
    data = pd.read_csv(data_path)
    logger.info("Loaded %d rows from %s", len(data), data_path)

    X = data[feature_cols].values
    X_processed = pipeline.transform(X)

    predictions, probabilities = predict_with_threshold(model, X_processed, threshold)

    results = pd.DataFrame(
        {
            "fraud_probability": probabilities,
            "is_fraud": predictions,
        }
    )

    if "txId" in data.columns:
        results.insert(0, "txId", data["txId"].values)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        logger.info("Saved predictions to %s", output_path)

    return results
