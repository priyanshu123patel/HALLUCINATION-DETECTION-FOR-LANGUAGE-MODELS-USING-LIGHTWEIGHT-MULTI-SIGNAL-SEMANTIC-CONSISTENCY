"""
Evaluation Metrics and Benchmark Reporting Module.

Computes academic classification metrics:
- Accuracy, Precision, Recall, F1-score, Macro F1
- ROC-AUC
- Confusion Matrix (TP, TN, FP, FN)
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Computational efficiency metrics (latency per sample, inference throughput)
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    latency_sec: float = 0.0
) -> Dict[str, Any]:
    """
    Computes rigorous evaluation metrics according to academic standards.
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    n_samples = len(y_true)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # False Positive Rate and False Negative Rate
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # ROC-AUC if probability estimates are provided
    roc_auc = None
    if y_prob is not None:
        try:
            y_prob_arr = np.array(y_prob)
            if y_prob_arr.ndim > 1 and y_prob_arr.shape[1] > 1:
                y_prob_arr = y_prob_arr[:, 1]
            roc_auc = float(roc_auc_score(y_true, y_prob_arr))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            roc_auc = None

    per_sample_latency_ms = (latency_sec / n_samples * 1000) if (n_samples > 0 and latency_sec > 0) else 0.0

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "total_samples": n_samples,
        "total_latency_seconds": round(latency_sec, 4),
        "latency_per_sample_ms": round(per_sample_latency_ms, 2)
    }

    return metrics


def save_metrics(metrics: Dict[str, Any], filepath: str):
    """
    Saves metrics dictionary to a JSON file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {path}")


def print_metrics_table(benchmark_results: Dict[str, Dict[str, Any]]):
    """
    Prints comparative metrics table to standard output.
    """
    rows = []
    for model_name, m in benchmark_results.items():
        rows.append({
            "Method / Model": model_name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A",
            "FPR": f"{m['false_positive_rate']:.4f}",
            "FNR": f"{m['false_negative_rate']:.4f}",
            "Latency (ms/sample)": f"{m['latency_per_sample_ms']:.1f}"
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("EXPERIMENTAL EVALUATION SUMMARY TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")
