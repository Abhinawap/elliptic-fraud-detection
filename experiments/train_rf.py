"""Train Random Forest fraud detector on Elliptic++ (run with --help for options).

Logs Gini feature importances as a CSV artifact to MLflow.
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src importable when running as a script without pip install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_elliptic_data, merge_and_filter_labeled
from src.features.feature_engineering import add_missingness_indicator, compute_graph_features
from src.models.evaluate import evaluate_classifier, optimize_threshold
from src.models.train import train_random_forest_with_gini
from src.preprocessing.pipeline import (
    create_preprocessing_pipeline,
    prepare_features_and_labels,
    temporal_train_test_split,
)
from src.utils.config import (
    DATA_DIR,
    EXPERIMENT_NAME,
    RF_182_OPTIMAL_THRESHOLD,
    RF_PARAMS,
    SPLIT_RATIO,
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
        description="Train Random Forest fraud detector on Elliptic++ dataset.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_mlflow_experiment(args.experiment_name)

    logger.info("Loading data from %s ...", args.data_dir)
    classes, features, edgelist, has_graph = load_elliptic_data(args.data_dir)
    _, labeled = merge_and_filter_labeled(features, classes)

    # missingness flag must come before imputation
    labeled = add_missingness_indicator(labeled)

    if has_graph and edgelist is not None:
        labeled = compute_graph_features(labeled, edgelist)
    else:
        logger.warning("No edgelist loaded; skipping graph features.")

    X, y, time_steps, feature_cols = prepare_features_and_labels(labeled)

    X_train, X_test, y_train, y_test, _, _ = temporal_train_test_split(
        X, y, time_steps, split_ratio=args.split_ratio
    )

    # fit only on train
    pipeline = create_preprocessing_pipeline()
    X_train_np = pipeline.fit_transform(X_train)
    X_test_np = pipeline.transform(X_test)

    # RF uses original {1, 2} labels (not converted to binary)
    rf, y_pred, y_proba, gini_importance = train_random_forest_with_gini(
        X_train_np,
        y_train.values,
        X_test_np,
        y_test.values,
        feature_cols,
        **RF_PARAMS,
    )

    # Evaluate — convert to binary {0,1} for metrics; RF trains on {1,2} which is valid
    # (illicit=1→1, licit=2→0); y_proba[:,0] is already P(illicit) so pos_label=1 is correct
    y_test_bin = (y_test == 1).astype(int).values
    y_pred_bin = (y_pred == 1).astype(int)
    metrics_default = evaluate_classifier(y_test_bin, y_pred_bin, y_proba, pos_label=1)
    optimal_threshold, y_pred_opt, metrics_opt = optimize_threshold(
        y_test_bin, y_proba, pos_label=1
    )

    logger.info(
        "Default threshold → F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f",
        metrics_default["f1_fraud"],
        metrics_default["roc_auc"],
        metrics_default["precision"],
        metrics_default["recall"],
    )
    logger.info(
        "Optimal threshold (%.4f) → F1=%.4f, AUC=%.4f, Precision=%.4f, Recall=%.4f",
        optimal_threshold,
        metrics_opt["f1_fraud"],
        metrics_opt["roc_auc"],
        metrics_opt["precision"],
        metrics_opt["recall"],
    )

    # Save Gini importance as CSV artifact
    importance_path = Path("models/checkpoints/rf_gini_importance.csv")
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    gini_importance.to_csv(importance_path, index=False)
    logger.info("Saved Gini importance to %s", importance_path)

    log_model_to_mlflow(
        model=rf,
        model_name="RF-182",
        params={
            **RF_PARAMS,
            "split_ratio": args.split_ratio,
            "n_features": len(feature_cols),
            "optimal_threshold": optimal_threshold,
        },
        metrics={
            **{f"default_{k}": v for k, v in metrics_default.items()},
            **{f"optimal_{k}": v for k, v in metrics_opt.items()},
        },
        X_train=X_train_np,
        artifacts=[str(importance_path)],
    )

    logger.info(
        "Done. Best F1=%.4f at threshold=%.4f. Run `mlflow ui` to view results.",
        metrics_opt["f1_fraud"],
        optimal_threshold,
    )


if __name__ == "__main__":
    main()
