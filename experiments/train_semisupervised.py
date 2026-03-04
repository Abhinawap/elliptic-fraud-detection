"""Semi-supervised XGBoost using pseudo-labels from unlabeled transactions.

Trains a base XGB-186, pseudo-labels high-confidence unlabeled rows, augments
the training set, and retrains. Results logged to MLflow alongside the base run.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_elliptic_data, merge_and_filter_labeled
from src.features.feature_engineering import (
    add_missingness_indicator,
    compute_graph_features,
    pseudo_label_unlabeled,
)
from src.models.evaluate import evaluate_classifier, optimize_threshold
from src.models.train import train_classifier
from src.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_labels,
    temporal_train_test_split,
)
from src.utils.config import (
    DATA_DIR,
    EXPERIMENT_NAME,
    SPLIT_RATIO,
    XGB_PARAMS,
)
from src.utils.mlflow_utils import log_model_to_mlflow, setup_mlflow_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semi-supervised XGBoost with pseudo-labels on unlabeled transactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to directory containing raw CSV files.",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=SPLIT_RATIO,
        help="Temporal train/test split ratio.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=EXPERIMENT_NAME,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--thresh-fraud",
        type=float,
        default=0.97,
        help="Minimum probability to pseudo-label a transaction as illicit.",
    )
    parser.add_argument(
        "--thresh-licit",
        type=float,
        default=0.03,
        help="Maximum probability to pseudo-label a transaction as licit.",
    )
    return parser.parse_args()


def _prepare_labeled(features, classes, edgelist, has_graph):
    """Apply missingness + graph features and return labeled subset."""
    _, labeled = merge_and_filter_labeled(features, classes)
    labeled = add_missingness_indicator(labeled)
    if has_graph and edgelist is not None:
        labeled = compute_graph_features(labeled, edgelist)
    else:
        logger.warning("No edgelist loaded; skipping graph features.")
    return labeled


def _prepare_unlabeled(features, classes, edgelist, has_graph):
    """Apply missingness + graph features and return unlabeled subset."""
    import pandas as pd
    from src.utils.config import UNKNOWN_LABEL

    merged = features.merge(classes[["txId", "class"]], on="txId", how="left")
    unlabeled = merged[merged["class"] == UNKNOWN_LABEL].copy()

    unlabeled = add_missingness_indicator(unlabeled)
    if has_graph and edgelist is not None:
        unlabeled = compute_graph_features(unlabeled, edgelist)
    logger.info("Unlabeled transactions available: %d", len(unlabeled))
    return unlabeled


def main() -> None:
    args = parse_args()
    setup_mlflow_experiment(args.experiment_name)

    logger.info("Loading data from %s ...", args.data_dir)
    classes, features, edgelist, has_graph = load_elliptic_data(args.data_dir)

    labeled = _prepare_labeled(features, classes, edgelist, has_graph)
    unlabeled = _prepare_unlabeled(features, classes, edgelist, has_graph)

    X_labeled, y_labeled, time_steps, feature_cols = prepare_features_and_labels(labeled)

    X_train, X_test, y_train, y_test, _, _ = temporal_train_test_split(
        X_labeled, y_labeled, time_steps, split_ratio=args.split_ratio
    )

    # fit pipeline on labeled train only
    pipeline = create_preprocessing_pipeline()
    X_train_np = pipeline.fit_transform(X_train)
    X_test_np = pipeline.transform(X_test)

    y_train_bin = (y_train == 1).astype(int).values
    y_test_bin = (y_test == 1).astype(int).values

    logger.info("Training base XGB-186 on labeled data (%d rows)...", len(y_train_bin))
    base_model = xgb.XGBClassifier(**XGB_PARAMS)
    base_model, _, base_proba = train_classifier(
        base_model, X_train_np, y_train_bin, X_test_np, y_test_bin, "XGB-186-base"
    )
    _, _, base_metrics = optimize_threshold(y_test_bin, base_proba, pos_label=1)
    logger.info(
        "Base model → F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f",
        base_metrics["f1_fraud"],
        base_metrics["roc_auc"],
        base_metrics["precision"],
        base_metrics["recall"],
    )

    if len(unlabeled) == 0:
        logger.warning("No unlabeled transactions found — skipping pseudo-labelling.")
        return

    X_unlabeled, _, _, _ = prepare_features_and_labels(unlabeled)
    X_unlabeled_np = pipeline.transform(X_unlabeled)

    X_pseudo, y_pseudo = pseudo_label_unlabeled(
        base_model,
        X_unlabeled_np,
        thresh_fraud=args.thresh_fraud,
        thresh_licit=args.thresh_licit,
    )
    logger.info(
        "Pseudo-labels: %d illicit, %d licit (from %d unlabeled)",
        int((y_pseudo == 1).sum()),
        int((y_pseudo == 0).sum()),
        len(X_unlabeled_np),
    )

    if len(X_pseudo) == 0:
        logger.warning(
            "No pseudo-labels generated at thresh_fraud=%.2f / thresh_licit=%.2f. "
            "Try relaxing confidence thresholds.",
            args.thresh_fraud,
            args.thresh_licit,
        )
        return

    X_aug = np.vstack([X_train_np, X_pseudo])
    y_aug = np.concatenate([y_train_bin, y_pseudo])
    logger.info(
        "Augmented training set: %d rows (%d original + %d pseudo-labeled)",
        len(y_aug),
        len(y_train_bin),
        len(y_pseudo),
    )

    semi_model = xgb.XGBClassifier(**XGB_PARAMS)
    semi_model, y_pred_semi, y_proba_semi = train_classifier(
        semi_model, X_aug, y_aug, X_test_np, y_test_bin, "XGB-186-semisupervised"
    )

    metrics_default = evaluate_classifier(y_test_bin, y_pred_semi, y_proba_semi, pos_label=1)
    optimal_threshold, _, metrics_opt = optimize_threshold(y_test_bin, y_proba_semi, pos_label=1)

    logger.info(
        "Semi-supervised (default) → F1=%.4f, AUC=%.4f",
        metrics_default["f1_fraud"],
        metrics_default["roc_auc"],
    )
    logger.info(
        "Semi-supervised (optimal threshold=%.4f) → F1=%.4f, AUC=%.4f, "
        "Precision=%.4f, Recall=%.4f",
        optimal_threshold,
        metrics_opt["f1_fraud"],
        metrics_opt["roc_auc"],
        metrics_opt["precision"],
        metrics_opt["recall"],
    )
    logger.info(
        "F1 delta vs base: %+.4f | Recall delta: %+.4f | Precision delta: %+.4f",
        metrics_opt["f1_fraud"] - base_metrics["f1_fraud"],
        metrics_opt["recall"] - base_metrics["recall"],
        metrics_opt["precision"] - base_metrics["precision"],
    )

    log_model_to_mlflow(
        model=semi_model,
        model_name="XGB-186-semisupervised",
        params={
            **XGB_PARAMS,
            "split_ratio": args.split_ratio,
            "n_features": len(feature_cols),
            "thresh_fraud": args.thresh_fraud,
            "thresh_licit": args.thresh_licit,
            "n_pseudo_labels": len(X_pseudo),
            "n_pseudo_illicit": int((y_pseudo == 1).sum()),
            "n_pseudo_licit": int((y_pseudo == 0).sum()),
            "optimal_threshold": optimal_threshold,
        },
        metrics={
            **{f"default_{k}": v for k, v in metrics_default.items()},
            **{f"optimal_{k}": v for k, v in metrics_opt.items()},
            "f1_delta_vs_base": metrics_opt["f1_fraud"] - base_metrics["f1_fraud"],
        },
        X_train=X_aug,
    )

    logger.info(
        "Done. Semi-supervised F1=%.4f at threshold=%.4f. Run `mlflow ui` to view results.",
        metrics_opt["f1_fraud"],
        optimal_threshold,
    )


if __name__ == "__main__":
    main()
