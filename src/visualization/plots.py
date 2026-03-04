"""
Visualization utilities for fraud detection analysis.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import DOCS_DIR, TRAIN_TEST_SPLIT_TIMESTEP

logger = logging.getLogger(__name__)


def plot_class_distribution(
    classes: pd.DataFrame,
    labeled: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """
    Create a 3-panel class distribution visualization.

    Args:
        classes: DataFrame with all transaction classes (including unknown).
        labeled: DataFrame with labeled transactions only (class 1 or 2).
        save_path: Path to save the plot. Defaults to DOCS_DIR/01_class_distribution.png.
    """
    if save_path is None:
        save_path = DOCS_DIR / "01_class_distribution.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fraud_count = int((labeled["class"] == 1).sum())
    legit_count = int((labeled["class"] == 2).sum())
    imbalance_ratio = legit_count / max(fraud_count, 1)

    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # Panel 1: All classes bar chart
    class_counts = classes["class"].value_counts()
    class_counts.plot(kind="bar", ax=ax1, color=["red", "green", "gray"], alpha=0.7)
    ax1.set_title("Class Distribution", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Class", fontsize=9)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.set_xticklabels(["Illicit", "Licit", "Unknown"], rotation=45, fontsize=8)

    # Panel 2: Labeled only pie chart
    labeled_counts = labeled["class"].value_counts().sort_index()
    pie_values = [labeled_counts.get(1, 0), labeled_counts.get(2, 0)]
    pie_labels = [f"Illicit\n{pie_values[0]:,}", f"Licit\n{pie_values[1]:,}"]
    ax2.pie(
        pie_values,
        labels=pie_labels,
        autopct="%1.1f%%",
        colors=["red", "green"],
        textprops={"fontsize": 8},
    )
    ax2.set_title("Labeled Only", fontsize=11, fontweight="bold")

    # Panel 3: Imbalance bar
    ax3.barh(
        ["Fraud", "Legit"],
        [fraud_count, legit_count],
        color=["red", "green"],
        alpha=0.7,
    )
    ax3.set_title(f"Imbalance: {imbalance_ratio:.1f}:1", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Count", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=80)
    plt.show()
    plt.close()
    logger.info("Saved plot to %s", save_path)


def plot_temporal_patterns(
    labeled: pd.DataFrame,
    edgelist: pd.DataFrame | None,
    train_test_split_point: int = TRAIN_TEST_SPLIT_TIMESTEP,
    save_path: Path | None = None,
) -> None:
    """
    Plot fraud rate and transaction volume over time.

    Args:
        labeled: DataFrame with labeled transactions (must have 'Time step' and 'class').
        edgelist: Transaction edge list (unused; kept for API compatibility).
        train_test_split_point: Timestep where train/test split occurs.
        save_path: Path to save the plot. Defaults to DOCS_DIR/02_temporal_patterns.png.
    """
    if save_path is None:
        save_path = DOCS_DIR / "02_temporal_patterns.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Fraud rate over time
    fraud_rate_by_time = labeled.groupby("Time step").apply(
        lambda x: (x["class"] == 1).mean()
    )
    axes[0].plot(
        fraud_rate_by_time.index,
        fraud_rate_by_time.values,
        marker="o",
        linewidth=2,
        markersize=4,
        color="darkred",
    )
    axes[0].set_title("Fraud Rate Over Time", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Fraud Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(
        x=train_test_split_point, color="gray", linestyle="--", label="Train/Test Split"
    )
    axes[0].legend()

    # Plot 2: Transaction volume over time
    volume_by_time = labeled.groupby("Time step").size()
    axes[1].plot(
        volume_by_time.index,
        volume_by_time.values,
        marker="o",
        color="steelblue",
        linewidth=2,
        markersize=4,
    )
    axes[1].set_title("Transaction Volume Over Time", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Transaction Count")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(
        x=train_test_split_point, color="gray", linestyle="--", label="Train/Test Split"
    )
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    logger.info("Saved plot to %s", save_path)


def plot_shap_analysis(
    shap_values_fraud: np.ndarray,
    shap_importance: pd.DataFrame,
    feature_cols: list[str],
    save_path: Path | None = None,
) -> None:
    """
    Create a two-panel SHAP analysis plot (magnitude + direction).

    Args:
        shap_values_fraud: SHAP values for the fraud class, shape (n_samples, n_features).
        shap_importance: DataFrame with columns ['feature', 'shap_importance'].
        feature_cols: Feature names (used to index shap_importance).
        save_path: Path to save the plot. Defaults to DOCS_DIR/03_shap_analysis.png.
    """
    if save_path is None:
        save_path = DOCS_DIR / "03_shap_analysis.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel 1: Top 15 features by mean |SHAP|
    top_shap = shap_importance.head(15)
    axes[0].barh(
        range(len(top_shap)),
        top_shap["shap_importance"].values,
        color="darkorange",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_yticks(range(len(top_shap)))
    axes[0].set_yticklabels(top_shap["feature"].values, fontsize=10)
    axes[0].set_xlabel("Mean |SHAP Value|", fontsize=11, fontweight="bold")
    axes[0].set_title(
        "Top 15 Features by SHAP Importance", fontsize=12, fontweight="bold", pad=10
    )
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis="x", linestyle="--")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    for i, v in enumerate(top_shap["shap_importance"].values):
        axes[0].text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=8)

    # Panel 2: Directional SHAP analysis
    top_15_idx = shap_importance.head(15).index
    shap_pos = (shap_values_fraud[:, top_15_idx] > 0).sum(axis=0)
    shap_neg = (shap_values_fraud[:, top_15_idx] < 0).sum(axis=0)

    x = np.arange(15)
    axes[1].barh(x, shap_pos, color="#d62728", alpha=0.8, label="Positive (→Fraud)",
                 edgecolor="black", linewidth=0.5)
    axes[1].barh(x, -shap_neg, color="#2ca02c", alpha=0.8, label="Negative (→Legit)",
                 edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(shap_importance.head(15)["feature"].values, fontsize=10)
    axes[1].set_xlabel("Count of SHAP Values", fontsize=11, fontweight="bold")
    axes[1].set_title(
        "SHAP Direction: Fraud vs Legit Impact", fontsize=12, fontweight="bold", pad=10
    )
    axes[1].invert_yaxis()
    axes[1].legend(loc="lower right", fontsize=10, framealpha=0.9)
    axes[1].axvline(x=0, color="black", linestyle="-", linewidth=1.2)
    axes[1].grid(True, alpha=0.3, axis="x", linestyle="--")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info("Saved plot to %s", save_path)


def plot_precision_recall_curves(
    models_dict: dict[str, dict],
    save_path: Path | None = None,
) -> None:
    """
    Plot precision-recall curves for multiple models side by side.

    Args:
        models_dict: Dictionary keyed by model name. Each value must contain:
            precision (array), recall (array), optimal_idx (int),
            optimal_threshold (float), and optionally color (str).
        save_path: Path to save the plot. Defaults to DOCS_DIR/04_precision_recall_tradeoff.png.
    """
    if save_path is None:
        save_path = DOCS_DIR / "04_precision_recall_tradeoff.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (model_name, model_data) in enumerate(models_dict.items()):
        precision = model_data["precision"]
        recall = model_data["recall"]
        optimal_idx = model_data["optimal_idx"]
        optimal_threshold = model_data["optimal_threshold"]
        color = model_data.get("color", "blue")

        axes[idx].plot(recall, precision, linewidth=2, color=color, label=model_name)
        axes[idx].scatter(
            recall[optimal_idx],
            precision[optimal_idx],
            s=100,
            c="red",
            marker="*",
            label=f"Optimal (t={optimal_threshold:.2f})",
            zorder=3,
        )
        axes[idx].set_xlabel("Recall (Fraud Detection Rate)", fontsize=11)
        axes[idx].set_ylabel("Precision (Alert Accuracy)", fontsize=11)
        axes[idx].set_title(
            f"{model_name}: Precision-Recall Trade-off", fontsize=12, fontweight="bold"
        )
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.show()
    plt.close()
    logger.info("Saved plot to %s", save_path)


def plot_learning_curve(
    learning_curve_df: pd.DataFrame,
    rf_baseline_f1: float,
    rf_baseline_auc: float,
    save_path: Path | None = None,
) -> None:
    """
    Plot XGBoost feature count learning curve (F1 and AUC vs. number of features).

    Args:
        learning_curve_df: DataFrame with columns: n_features, fraud_f1_default,
            fraud_f1_optimized, fraud_recall, fraud_precision, roc_auc.
        rf_baseline_f1: RF-182 optimised F1 score (horizontal reference line).
        rf_baseline_auc: RF-182 ROC-AUC score (horizontal reference line).
        save_path: Path to save the plot. Defaults to DOCS_DIR/05_xgboost_learning_curve.png.
    """
    if save_path is None:
        save_path = DOCS_DIR / "05_xgboost_learning_curve.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: F1 vs features
    axes[0, 0].plot(
        learning_curve_df["n_features"],
        learning_curve_df["fraud_f1_default"],
        marker="o",
        label="Default Threshold",
        linewidth=2,
    )
    axes[0, 0].plot(
        learning_curve_df["n_features"],
        learning_curve_df["fraud_f1_optimized"],
        marker="s",
        label="Optimized Threshold",
        linewidth=2,
    )
    axes[0, 0].axhline(
        y=rf_baseline_f1, color="green", linestyle="--", label="RF-182 (Optimized)", alpha=0.7
    )
    axes[0, 0].set_xlabel("Number of Features")
    axes[0, 0].set_ylabel("Fraud F1 Score")
    axes[0, 0].set_title("XGBoost Learning Curve: F1 Score")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Precision-Recall scatter by feature count
    axes[0, 1].scatter(
        learning_curve_df["fraud_recall"],
        learning_curve_df["fraud_precision"],
        c=learning_curve_df["n_features"],
        cmap="viridis",
        s=100,
    )
    for i, n in enumerate(learning_curve_df["n_features"]):
        axes[0, 1].annotate(
            f"{int(n)}",
            (learning_curve_df["fraud_recall"].iloc[i], learning_curve_df["fraud_precision"].iloc[i]),
            fontsize=8,
        )
    axes[0, 1].set_xlabel("Fraud Recall")
    axes[0, 1].set_ylabel("Fraud Precision")
    axes[0, 1].set_title("Precision-Recall Trade-off by Feature Count")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: ROC-AUC vs features
    axes[1, 0].plot(
        learning_curve_df["n_features"],
        learning_curve_df["roc_auc"],
        marker="o",
        color="purple",
        linewidth=2,
    )
    axes[1, 0].axhline(
        y=rf_baseline_auc, color="green", linestyle="--", label="RF-182", alpha=0.7
    )
    axes[1, 0].set_xlabel("Number of Features")
    axes[1, 0].set_ylabel("ROC-AUC Score")
    axes[1, 0].set_title("XGBoost Learning Curve: ROC-AUC")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Threshold optimisation benefit
    learning_curve_df = learning_curve_df.copy()
    learning_curve_df["f1_improvement"] = (
        learning_curve_df["fraud_f1_optimized"] - learning_curve_df["fraud_f1_default"]
    )
    axes[1, 1].bar(
        range(len(learning_curve_df)),
        learning_curve_df["f1_improvement"],
        color="orange",
        alpha=0.7,
    )
    axes[1, 1].set_xticks(range(len(learning_curve_df)))
    axes[1, 1].set_xticklabels(learning_curve_df["n_features"].astype(int), rotation=45)
    axes[1, 1].set_xlabel("Number of Features")
    axes[1, 1].set_ylabel("F1 Improvement")
    axes[1, 1].set_title("Benefit of Threshold Optimization")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()
    logger.info("Saved plot to %s", save_path)
