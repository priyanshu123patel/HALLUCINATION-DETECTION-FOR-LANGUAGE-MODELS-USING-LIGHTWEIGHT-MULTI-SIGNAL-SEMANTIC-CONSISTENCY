"""
Academic Visualization Module for Hallucination Detection Research.

Generates publication-quality figures using matplotlib and seaborn:
1. class_distribution.png
2. baseline_comparison.png
3. precision_recall_f1_comparison.png
4. confusion_matrix.png
5. roc_curve.png
6. feature_importance.png
7. ablation_study.png
8. score_distribution.png
9. sentence_level_example.png
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

# Academic figure aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "figure.dpi": 300
})

PALETTE = {
    "supported": "#2b5c8f",     # Deep Academic Blue
    "hallucinated": "#d95f02",  # Vivid Coral / Orange
    "neutral": "#7570b3",
    "accent": "#1b9e77"
}


def ensure_dir(filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    save_path: str = "results/figures/class_distribution.png"
):
    """
    Plots the class balance across Train, Validation, and Test splits.
    """
    ensure_dir(save_path)
    splits = []
    labels = []
    counts = []

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        c0 = (df["label"] == 0).sum()
        c1 = (df["label"] == 1).sum()
        splits.extend([name, name])
        labels.extend(["SUPPORTED (0)", "HALLUCINATED (1)"])
        counts.extend([c0, c1])

    data = pd.DataFrame({"Split": splits, "Class": labels, "Count": counts})

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(
        data=data,
        x="Split",
        y="Count",
        hue="Class",
        palette=[PALETTE["supported"], PALETTE["hallucinated"]],
        ax=ax
    )

    # Add count annotations on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height / 2.0),
                ha="center", va="center", color="white", fontweight="bold", fontsize=10
            )

    ax.set_title("HaluEval Benchmark Dataset Split & Class Distribution", pad=15)
    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Number of Samples")
    ax.legend(title="Ground Truth Label", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved class distribution figure to {save_path}")


def plot_baseline_comparison(
    benchmark_df: pd.DataFrame,
    save_path: str = "results/figures/baseline_comparison.png"
):
    """
    Bar chart comparing Overall Accuracy and Macro F1 across all evaluated methods.
    """
    ensure_dir(save_path)
    melted = benchmark_df.melt(
        id_vars=["model"],
        value_vars=["accuracy", "macro_f1"],
        var_name="Metric",
        value_name="Score"
    )
    metric_map = {"accuracy": "Accuracy", "macro_f1": "Macro F1"}
    melted["Metric"] = melted["Metric"].map(metric_map)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=melted,
        x="model",
        y="Score",
        hue="Metric",
        palette=[PALETTE["supported"], PALETTE["accent"]],
        ax=ax
    )

    # Annotate bar values
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.015),
                ha="center", va="bottom", fontsize=9, fontweight="bold"
            )

    ax.set_ylim(0.0, 1.05)
    ax.set_title("Hallucination Detection: Baselines vs Proposed Detector", pad=15)
    ax.set_xlabel("Method / Baseline")
    ax.set_ylabel("Performance Score")
    ax.legend(title="Metric", frameon=True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved baseline comparison figure to {save_path}")


def plot_precision_recall_f1(
    benchmark_df: pd.DataFrame,
    save_path: str = "results/figures/precision_recall_f1_comparison.png"
):
    """
    Grouped bar chart comparing Precision, Recall, and F1 across methods.
    """
    ensure_dir(save_path)
    melted = benchmark_df.melt(
        id_vars=["model"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="Metric",
        value_name="Score"
    )
    name_map = {"precision": "Precision", "recall": "Recall", "f1_score": "F1-Score"}
    melted["Metric"] = melted["Metric"].map(name_map)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=melted,
        x="model",
        y="Score",
        hue="Metric",
        palette=["#386cb0", "#f0027f", "#7fc97f"],
        ax=ax
    )

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.2f}",
                (p.get_x() + p.get_width() / 2.0, h + 0.015),
                ha="center", va="bottom", fontsize=8
            )

    ax.set_ylim(0.0, 1.05)
    ax.set_title("Hallucination Detection: Precision, Recall & F1 Trade-off", pad=15)
    ax.set_xlabel("Model / Baseline")
    ax.set_ylabel("Score")
    ax.legend(title="Metric", frameon=True)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved precision/recall/f1 comparison to {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Proposed Detector",
    save_path: str = "results/figures/confusion_matrix.png"
):
    """
    Plots a normalized and annotated confusion matrix heatmap.
    """
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    annot = np.empty_like(cm).astype(str)
    for i in range(2):
        for j in range(2):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["SUPPORTED (0)", "HALLUCINATED (1)"],
        yticklabels=["SUPPORTED (0)", "HALLUCINATED (1)"],
        annot_kws={"fontsize": 12, "fontweight": "bold"}
    )

    ax.set_title(f"Confusion Matrix: {model_name}", pad=15)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("Ground Truth Label", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix figure to {save_path}")


def plot_roc_curve(
    roc_dict: Dict[str, Dict[str, Any]],
    save_path: str = "results/figures/roc_curve.png"
):
    """
    Overlays ROC curves and AUC scores for each evaluated baseline and proposed model.
    """
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#7570b3", "#e7298a", "#1b9e77", "#d95f02", "#386cb0"]
    for i, (name, data) in enumerate(roc_dict.items()):
        fpr = data["fpr"]
        tpr = data["tpr"]
        roc_auc = data["auc"]
        color = colors[i % len(colors)]
        ax.plot(
            fpr, tpr,
            label=f"{name} (AUC = {roc_auc:.3f})",
            lw=2.2,
            color=color
        )

    # Reference random line
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Guess (AUC = 0.500)")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", pad=15)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve figure to {save_path}")


def plot_feature_importance(
    feature_df: pd.DataFrame,
    save_path: str = "results/figures/feature_importance.png"
):
    """
    Plots horizontal bar chart of learned Logistic Regression coefficients.
    Clearly distinguishes features pushing towards hallucination (+) vs support (-).
    """
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, 6))
    df = feature_df.sort_values("coefficient", ascending=True).copy()

    colors = [PALETTE["hallucinated"] if c > 0 else PALETTE["supported"] for c in df["coefficient"]]

    bars = ax.barh(df["feature"], df["coefficient"], color=colors, edgecolor="black", alpha=0.85)

    # Zero line
    ax.axvline(0, color="black", linestyle="--", lw=1)

    for bar in bars:
        w = bar.get_width()
        offset = 0.05 if w >= 0 else -0.05
        ha = "left" if w >= 0 else "right"
        ax.annotate(
            f"{w:+.2f}",
            (w + offset, bar.get_y() + bar.get_height() / 2.0),
            va="center", ha=ha, fontsize=9, fontweight="bold"
        )

    ax.set_title("Learned Feature Contributions (Standardized Logistic Regression)", pad=15)
    ax.set_xlabel("Coefficient Weight (Positive = Hallucination, Negative = Supported)")
    ax.set_ylabel("Extracted Signal Feature")

    # Custom legend
    import matplotlib.patches as mpatches
    p1 = mpatches.Patch(color=PALETTE["hallucinated"], label="Pushes Toward Hallucination (+)")
    p2 = mpatches.Patch(color=PALETTE["supported"], label="Pushes Toward Supported (-)")
    ax.legend(handles=[p1, p2], loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance figure to {save_path}")


def plot_ablation_study(
    ablation_df: pd.DataFrame,
    save_path: str = "results/figures/ablation_study.png"
):
    """
    Visualizes the ablation study results: performance impact when feature groups are removed.
    """
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(9, 5))

    df = ablation_df.sort_values("f1_score", ascending=True)
    bars = ax.barh(
        df["configuration"],
        df["f1_score"],
        color=["#e41a1c" if "Full" not in c else "#377eb8" for c in df["configuration"]],
        edgecolor="black",
        alpha=0.85
    )

    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f"F1: {w:.3f}",
            (w - 0.08, bar.get_y() + bar.get_height() / 2.0),
            va="center", ha="right", color="white", fontsize=10, fontweight="bold"
        )

    ax.set_xlim(0.0, 1.05)
    ax.set_title("Ablation Study: Feature Group Contribution to Detection F1", pad=15)
    ax.set_xlabel("Test Set F1-Score")
    ax.set_ylabel("Configuration")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ablation study figure to {save_path}")


def plot_score_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str = "results/figures/score_distribution.png"
):
    """
    Visualizes KDE / distribution of predicted hallucination probabilities for both ground-truth classes.
    """
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    df = pd.DataFrame({
        "Ground Truth": ["HALLUCINATED" if y == 1 else "SUPPORTED" for y in y_true],
        "Hallucination Probability": y_prob
    })

    sns.kdeplot(
        data=df,
        x="Hallucination Probability",
        hue="Ground Truth",
        common_norm=False,
        fill=True,
        alpha=0.4,
        palette=[PALETTE["hallucinated"], PALETTE["supported"]],
        ax=ax
    )

    ax.axvline(0.50, color="gray", linestyle="--", lw=1.5, label="Decision Threshold (0.50)")
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Distribution of Predicted Hallucination Probabilities", pad=15)
    ax.set_xlabel("Predicted Probability of Hallucination P(y = 1)")
    ax.set_ylabel("Density")
    ax.legend(title="Ground Truth", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved score distribution figure to {save_path}")


def plot_sentence_level_example(
    sentence_records: List[Dict[str, Any]],
    save_path: str = "results/figures/sentence_level_example.png"
):
    """
    Generates a visual card / bar diagram of sentence-level suspiciousness scores for research demonstration.
    """
    ensure_dir(save_path)
    fig, ax = plt.subplots(figsize=(10, len(sentence_records) * 1.2 + 2))

    labels = [f"Sentence {i+1}:\n\"{r['sentence'][:40]}...\"" for i, r in enumerate(sentence_records)]
    contr_scores = [r.get("contradiction_prob", 0.0) for r in sentence_records]
    sem_scores = [r.get("semantic_similarity", 0.0) for r in sentence_records]
    statuses = [r.get("status", "SUPPORTED") for r in sentence_records]

    y_pos = np.arange(len(sentence_records))
    height = 0.35

    ax.barh(y_pos - height / 2, contr_scores, height, label="Contradiction Probability", color=PALETTE["hallucinated"], alpha=0.85)
    ax.barh(y_pos + height / 2, sem_scores, height, label="Semantic Similarity", color=PALETTE["supported"], alpha=0.85)

    for i, status in enumerate(statuses):
        color = "#e41a1c" if status == "SUSPICIOUS" else "#4daf4a"
        ax.annotate(
            f"[{status}]",
            (1.02, y_pos[i]),
            va="center", ha="left", color=color, fontweight="bold", fontsize=10
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.25)
    ax.set_xlabel("Signal Score (0.0 to 1.0)")
    ax.set_title("Sentence-Level Hallucination Diagnostic Analysis", pad=15)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved sentence-level example figure to {save_path}")
