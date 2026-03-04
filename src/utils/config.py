"""Hyperparameters and constants for the Elliptic++ fraud detection project."""

from pathlib import Path

RANDOM_STATE: int = 42

# paths
DATA_DIR: Path = Path("data/raw")
DOCS_DIR: Path = Path("docs/images")
MLFLOW_DIR: Path = Path("mlruns")
MODELS_DIR: Path = Path("models/checkpoints")

SPLIT_RATIO: float = 0.8
EXCLUDE_COLS: list[str] = ["txId", "class", "Time step"]

# Timesteps 1-41 train, 42-49 test (80/20 split by time)
TRAIN_TEST_SPLIT_TIMESTEP: int = 41

SHAP_SAMPLE_SIZE: int = 5000

EXPERIMENT_NAME: str = "elliptic-fraud-detection"

# class labels: 1=illicit, 2=licit, 3=unknown
ILLICIT_LABEL: int = 1
LICIT_LABEL: int = 2
UNKNOWN_LABEL: int = 3

# Computed from labelled data: 42,019 licit / 4,545 illicit ≈ 9.25
CLASS_IMBALANCE_RATIO: float = 9.25

# RF params
RF_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# XGB params
XGB_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": CLASS_IMBALANCE_RATIO,  # upweights fraud class
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

# thresholds tuned in notebooks
XGB_182_OPTIMAL_THRESHOLD: float = 0.910  # updated: 186-feature model (+ missingness + graph)
RF_182_OPTIMAL_THRESHOLD: float = 0.770  # updated: 186-feature model (+ missingness + graph)

# These 17 columns are NaN only for licit transactions (0/4545 illicit rows
# have missing values; 519/42019 licit rows do). Must be captured BEFORE
# the SimpleImputer step in the preprocessing pipeline.
STRUCTURAL_MISSING_COLS: list[str] = [
    "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
    "num_input_addresses", "num_output_addresses",
    "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
    "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total",
]

# Contamination ≈ true illicit rate (9.76%) + small margin
ISOLATION_FOREST_PARAMS: dict = {
    "n_estimators": 100,
    "contamination": 0.10,
    "max_samples": "auto",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
