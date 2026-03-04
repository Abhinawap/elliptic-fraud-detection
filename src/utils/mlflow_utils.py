"""
MLflow logging utilities for model tracking and experiment management.
"""

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.utils.config import EXPERIMENT_NAME

logger = logging.getLogger(__name__)


def log_model_to_mlflow(
    model: Any,
    model_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    X_train: NDArray | pd.DataFrame,
    artifacts: list[str] | None = None,
) -> None:
    """
    Log a trained model, its parameters, metrics, and optional artifacts to MLflow.

    Args:
        model: Fitted model instance (RandomForestClassifier or XGBClassifier).
        model_name: Display name for the MLflow run.
        params: Hyperparameters to log.
        metrics: Evaluation metrics to log.
        X_train: Training data used to infer the model signature.
        artifacts: Optional list of local file paths to log as MLflow artifacts.
    """
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    from mlflow.models.signature import infer_signature

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, model.predict(X_train))

        # Use XGBoost-native logging if available; otherwise fall back to sklearn
        try:
            import xgboost

            if isinstance(model, xgboost.XGBModel):
                mlflow.xgboost.log_model(model, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)
        except ImportError:
            mlflow.sklearn.log_model(model, "model", signature=signature)

        if artifacts:
            for artifact_path in artifacts:
                if os.path.exists(artifact_path):
                    mlflow.log_artifact(artifact_path)
                else:
                    logger.warning("Artifact not found, skipping: %s", artifact_path)

    logger.info(
        "MLflow run complete: model=%s, f1_fraud=%.4f",
        model_name,
        metrics.get("optimal_f1_fraud", metrics.get("f1_fraud", float("nan"))),
    )


def setup_mlflow_experiment(
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """
    Initialise (or retrieve) a named MLflow experiment.

    Args:
        experiment_name: Human-readable experiment name.

    Returns:
        experiment_id: MLflow experiment ID string.
    """
    import mlflow

    mlflow.set_experiment(experiment_name)
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    logger.info("MLflow experiment: '%s' (id=%s)", experiment_name, experiment_id)
    return experiment_id
