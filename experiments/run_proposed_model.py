"""
Experiment 4: Proposed Lightweight Multi-Signal Hallucination Detector.

Combines:
- TF-IDF similarity + Jaccard token overlap (Lexical)
- MiniLM-L6-v2 dense embeddings (Semantic)
- NLI entailment, contradiction, and neutral signals (Natural Language Inference)
- Named Entity consistency metrics (spaCy NER)
- Sentence-level anomaly signals (Sentence Consistency)

Trains and evaluates:
A. Primary Learned Model (Logistic Regression with Standard Scaler)
B. Configurable Heuristic Weighted Score (Formula-based baseline for ablation)
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

from src.data_loader import HaluEvalDataLoader, load_config
from src.feature_extraction import FeatureExtractor, ALL_FEATURE_COLUMNS
from src.detector import LightweightHallucinationDetector
from src.evaluation import compute_metrics, save_metrics, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_proposed_model(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Trains and benchmarks the proposed multi-signal hallucination detector.
    """
    config = load_config(config_path)
    loader = HaluEvalDataLoader(config=config)
    train_raw, val_raw, test_raw = loader.load_or_prepare_splits()

    extractor = FeatureExtractor(config=config)
    train_feat = extractor.extract_dataset_features(train_raw, split_name="train")
    test_feat = extractor.extract_dataset_features(test_raw, split_name="test")

    feature_cols = [c for c in ALL_FEATURE_COLUMNS if c in train_feat.columns]
    logger.info(f"Training proposed detector with {len(feature_cols)} features: {feature_cols}")

    detector = LightweightHallucinationDetector(
        feature_columns=feature_cols,
        config=config,
        C=config.get("classifier", {}).get("C", 1.0),
        class_weight=config.get("classifier", {}).get("class_weight", "balanced")
    )

    # 1. Fit learned model
    detector.fit(train_feat[feature_cols], train_feat["label"])
    detector.save("models/hallucination_detector.joblib")

    # 2. Evaluate Learned Model on Test Set
    t0 = time.time()
    y_prob = detector.predict_proba(test_feat[feature_cols])[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)
    latency = time.time() - t0

    learned_metrics = compute_metrics(
        test_feat["label"].values,
        y_pred,
        y_prob,
        latency_sec=latency
    )

    # 3. Evaluate Method B: Configurable Heuristic Weighted Score
    w_probs = []
    w_preds = []
    for _, row in test_feat.iterrows():
        score, pred = detector.compute_weighted_score(row.to_dict())
        w_probs.append(score)
        w_preds.append(pred)

    weighted_metrics = compute_metrics(
        test_feat["label"].values,
        w_preds,
        w_probs,
        latency_sec=0.01
    )

    results = {
        "Proposed Method: Learned Multi-Signal Detector": learned_metrics,
        "Proposed Method: Configurable Weighted Score (Heuristic)": weighted_metrics
    }

    # Save metrics and predictions
    save_metrics(results, "results/metrics/proposed_metrics.json")

    # Export predictions dataframe
    preds_df = pd.DataFrame({
        "context": test_feat["context"],
        "answer": test_feat["answer"],
        "ground_truth": test_feat["label"],
        "learned_prob": np.round(y_prob, 4),
        "learned_pred": y_pred,
        "heuristic_score": np.round(w_probs, 4),
        "heuristic_pred": w_preds
    })
    for col in feature_cols:
        preds_df[col] = test_feat[col]

    preds_df.to_csv("results/predictions/proposed_predictions.csv", index=False)
    print_metrics_table(results)

    # Print feature importances
    fi_df = detector.get_feature_importances()
    print("\n" + "=" * 70)
    print("LEARNED FEATURE IMPORTANCES & WEIGHTS (Proposed Detector)")
    print("=" * 70)
    print(fi_df.to_string(index=False))
    print("=" * 70 + "\n")
    fi_df.to_csv("results/metrics/feature_importances.csv", index=False)

    return results


if __name__ == "__main__":
    run_proposed_model()
