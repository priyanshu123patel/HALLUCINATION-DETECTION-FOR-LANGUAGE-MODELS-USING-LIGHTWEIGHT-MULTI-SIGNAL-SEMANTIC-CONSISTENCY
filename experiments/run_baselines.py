"""
Experiment 1, 2, 3: Baseline Hallucination Detection Models.

Evaluates independent baselines:
- Baseline 1: TF-IDF Lexical Similarity (Cosine similarity -> Logistic Regression)
- Baseline 2: Semantic Similarity (MiniLM-L6-v2 embeddings -> Logistic Regression)
- Baseline 3: NLI (Natural Language Inference probabilities -> Logistic Regression)
- Baseline 4: Lexical + Semantic Combination (TF-IDF + MiniLM -> Logistic Regression)

Produces standardized academic metrics:
- Accuracy, Precision, Recall, F1-Score, Macro F1, ROC-AUC, FPR, FNR, Latency
"""

import time
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data_loader import HaluEvalDataLoader, load_config
from src.feature_extraction import FeatureExtractor
from src.evaluation import compute_metrics, save_metrics, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_and_eval_baseline(
    baseline_name: str,
    feature_cols: list,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Fits standard scaler and logistic regression on specified feature columns.
    """
    logger.info(f"--- Running {baseline_name} (features: {feature_cols}) ---")
    scaler = StandardScaler()
    clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)

    X_train = scaler.fit_transform(train_df[feature_cols].values)
    y_train = train_df["label"].values.astype(int)

    X_test = scaler.transform(test_df[feature_cols].values)
    y_test = test_df["label"].values.astype(int)

    clf.fit(X_train, y_train)

    t0 = time.time()
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    latency = time.time() - t0

    metrics = compute_metrics(y_test, y_pred, y_prob, latency_sec=latency)
    logger.info(f"{baseline_name} -> Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | AUC: {metrics['roc_auc']}")

    return {
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_prob,
        "model": clf,
        "scaler": scaler
    }


def run_baselines(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Main runner for all baselines.
    """
    config = load_config(config_path)
    loader = HaluEvalDataLoader(config=config)
    train_raw, val_raw, test_raw = loader.load_or_prepare_splits()

    extractor = FeatureExtractor(config=config)
    train_feat = extractor.extract_dataset_features(train_raw, split_name="train")
    test_feat = extractor.extract_dataset_features(test_raw, split_name="test")

    results = {}
    preds_df = pd.DataFrame({
        "context": test_feat["context"],
        "answer": test_feat["answer"],
        "ground_truth": test_feat["label"]
    })

    # Baseline 1: TF-IDF
    b1_res = train_and_eval_baseline("Baseline 1: TF-IDF Similarity", ["tfidf_similarity"], train_feat, test_feat)
    results["Baseline 1: TF-IDF Similarity"] = b1_res["metrics"]
    preds_df["b1_tfidf_prob"] = b1_res["probabilities"]
    preds_df["b1_tfidf_pred"] = b1_res["predictions"]

    # Baseline 2: Semantic Similarity
    b2_res = train_and_eval_baseline("Baseline 2: Semantic Similarity", ["semantic_similarity"], train_feat, test_feat)
    results["Baseline 2: Semantic Similarity"] = b2_res["metrics"]
    preds_df["b2_semantic_prob"] = b2_res["probabilities"]
    preds_df["b2_semantic_pred"] = b2_res["predictions"]

    # Baseline 3: NLI (Natural Language Inference)
    nli_cols = ["nli_contradiction_prob", "nli_entailment_prob", "nli_neutral_prob"]
    b3_res = train_and_eval_baseline("Baseline 3: NLI (Entailment/Contradiction)", nli_cols, train_feat, test_feat)
    results["Baseline 3: NLI (Entailment/Contradiction)"] = b3_res["metrics"]
    preds_df["b3_nli_prob"] = b3_res["probabilities"]
    preds_df["b3_nli_pred"] = b3_res["predictions"]

    # Baseline 4: Lexical + Semantic
    b4_res = train_and_eval_baseline("Baseline 4: TF-IDF + Semantic", ["tfidf_similarity", "semantic_similarity"], train_feat, test_feat)
    results["Baseline 4: TF-IDF + Semantic"] = b4_res["metrics"]
    preds_df["b4_lex_sem_prob"] = b4_res["probabilities"]
    preds_df["b4_lex_sem_pred"] = b4_res["predictions"]

    # Save metrics and predictions
    save_metrics(results, "results/metrics/baselines_metrics.json")
    preds_df.to_csv("results/predictions/baselines_predictions.csv", index=False)
    print_metrics_table(results)

    return results


if __name__ == "__main__":
    run_baselines()
