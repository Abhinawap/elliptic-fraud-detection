"""Shared pytest fixtures — all use synthetic data, no CSV files required."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_features() -> pd.DataFrame:
    """
    Minimal synthetic features DataFrame mimicking txs_features.csv.

    245 transactions × 10 numeric features, plus txId and Time step columns.
    Spans timesteps 1–49 (5 tx each) so the fixed split boundary at timestep 41
    produces a non-empty test set (timesteps 42–49).
    Uses a fixed random seed so results are deterministic across runs.
    """
    rng = np.random.RandomState(42)
    n = 245
    data: dict = {
        "txId": np.arange(1, n + 1),
        "Time step": np.repeat(np.arange(1, 50), 5),
    }
    for i in range(1, 11):
        data[f"feature_{i}"] = rng.randn(n).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def synthetic_classes(synthetic_features: pd.DataFrame) -> pd.DataFrame:
    """
    Transaction labels mimicking txs_classes.csv.

    ~10% illicit (class=1), ~90% licit (class=2).
    All 200 transactions are labeled for simplicity (no unknown class 3).
    """
    rng = np.random.RandomState(42)
    n = len(synthetic_features)
    labels = rng.choice([1, 2], size=n, p=[0.1, 0.9])
    return pd.DataFrame({"txId": synthetic_features["txId"].values, "class": labels})


@pytest.fixture(scope="session")
def synthetic_labeled(
    synthetic_features: pd.DataFrame, synthetic_classes: pd.DataFrame
) -> pd.DataFrame:
    """
    Pre-merged labeled DataFrame (features + class labels joined on txId).
    """
    data = synthetic_features.merge(synthetic_classes, on="txId", how="left")
    data["class"] = data["class"].astype(int)
    return data


@pytest.fixture(scope="session")
def synthetic_edgelist() -> pd.DataFrame:
    """Minimal transaction edge list with 5 directed edges."""
    return pd.DataFrame(
        {
            "txId1": [1, 2, 3, 4, 5],
            "txId2": [2, 3, 4, 5, 6],
        }
    )


@pytest.fixture(scope="session")
def feature_cols(synthetic_features: pd.DataFrame) -> list[str]:
    """List of numeric feature column names (excludes txId and Time step)."""
    return [c for c in synthetic_features.columns if c.startswith("feature_")]
