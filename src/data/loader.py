"""
Data loading utilities for Elliptic++ dataset.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.config import DATA_DIR

logger = logging.getLogger(__name__)


def load_elliptic_data(
    data_dir: Path = DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None, bool]:
    """
    Load Elliptic++ dataset files.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        classes: txId + class columns.
        features: txId, Time step, and 182 numeric feature columns.
        edgelist: directed edges, or None if file not found.
        has_graph: True if edgelist was loaded successfully.

    Raises:
        FileNotFoundError: If data_dir does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Load classes
    classes = pd.read_csv(data_dir / "txs_classes.csv")
    logger.info("Loaded %d transaction class labels from %s", len(classes), data_dir)

    # Load features — first row is the header stored as data
    features = pd.read_csv(data_dir / "txs_features.csv", header=None)
    features.columns = features.iloc[0]
    features = features[1:].reset_index(drop=True)

    # Convert all columns to numeric
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    logger.info(
        "Loaded transaction features: %d rows × %d columns", len(features), features.shape[1]
    )

    # Validate txId consistency between classes and features
    consistency = validate_data_consistency(classes, features)
    if not consistency["is_consistent"]:
        logger.warning(
            "txId mismatch: classes=%d, features=%d, common=%d",
            consistency["classes_txids"],
            consistency["features_txids"],
            consistency["common_txids"],
        )

    # Load edgelist (optional — not required for classical ML)
    try:
        edgelist = pd.read_csv(data_dir / "txs_edgelist.csv")
        has_graph = True
        logger.info("Loaded %d transaction edges", len(edgelist))
    except FileNotFoundError:
        logger.warning(
            "txs_edgelist.csv not found in %s; graph features disabled.", data_dir
        )
        edgelist = None
        has_graph = False
    except pd.errors.ParserError as exc:
        logger.error("Failed to parse txs_edgelist.csv: %s", exc)
        raise

    return classes, features, edgelist, has_graph


def validate_data_consistency(
    classes: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, int | bool]:
    """Check txId overlap between classes and features DataFrames."""
    unique_txids_classes = classes["txId"].nunique()
    unique_txids_features = features["txId"].nunique()
    common_txids = len(set(classes["txId"]).intersection(set(features["txId"])))

    is_consistent = unique_txids_classes == unique_txids_features == common_txids
    if not is_consistent:
        logger.warning(
            "Data inconsistency detected: classes=%d, features=%d, common=%d",
            unique_txids_classes,
            unique_txids_features,
            common_txids,
        )

    return {
        "classes_txids": unique_txids_classes,
        "features_txids": unique_txids_features,
        "common_txids": common_txids,
        "is_consistent": is_consistent,
    }


def merge_and_filter_labeled(
    features: pd.DataFrame,
    classes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge features with classes and filter to labeled transactions only.

    Labeled transactions are those with class 1 (illicit) or 2 (licit).
    Transactions with class 3 (unknown) are excluded from the labeled set.

    Args:
        features: DataFrame with transaction features.
        classes: DataFrame with transaction class labels.

    Returns:
        data: Full merged DataFrame (all transactions, including unknown).
        labeled: Filtered DataFrame containing only labeled transactions.
    """
    data = features.merge(classes, on="txId", how="left")

    # Filter to labeled data — handle both string and numeric formats
    labeled = data[data["class"].isin([1, 2, "1", "2"])].copy()

    # Ensure class is stored as integer
    labeled["class"] = labeled["class"].astype(float).astype(int)

    illicit_count = (labeled["class"] == 1).sum()
    licit_count = (labeled["class"] == 2).sum()
    logger.info(
        "Labeled dataset: %d transactions (%d illicit, %d licit, imbalance ratio %.2f:1)",
        len(labeled),
        illicit_count,
        licit_count,
        licit_count / max(illicit_count, 1),
    )

    return data, labeled
