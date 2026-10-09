"""
Experiment 5: Feature Ablation Study.

Systematically evaluates the contribution of each signal component:
- Full Proposed Model (all features)
- Ablation 1: Without Lexical Features (TF-IDF, Jaccard overlap)
- Ablation 2: Without Semantic Features (MiniLM embeddings, sentence min/mean)
- Ablation 3: Without NLI Features (Contradiction, Entailment, Neutral)
- Ablation 4: Without Entity Features (Unsupported entity count/ratio, Jaccard)
- Ablation 5: Without Sentence-Level Features (Unsupported sentence ratio, length)

Calculates the performance drop (Delta F1, Delta Accuracy) to isolate feature necessity.
"""

import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data_loader import HaluEvalDataLoader, load_config
from src.feature_extraction import FeatureExtractor, FEATURE_GROUPS, ALL_FEATURE_COLUMNS
from src.evaluation import compute_metrics, save_metrics, print_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ablation(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Executes systematic leave-one-group-out ablation experiments.
    """
    config = load_config(config_path)
    loader = HaluEvalDataLoader(config=config)
    train_raw, val_raw, test_raw = loader.load_or_prepare_splits()

    extractor = FeatureExtractor(config=config)
    train_feat = extractor.extract_dataset_features(train_raw, split_name="train")
    test_feat = extractor.extract_dataset_features(test_raw, split_name="test")

    available_cols = [c for c in ALL_FEATURE_COLUMNS if c in train_feat.columns]

    # Define ablation configurations
    configurations = {
        "Full Proposed Model": available_cols,
        "w/o Lexical Features": [c for c in available_cols if c not in FEATURE_GROUPS["lexical"]],
        "w/o Semantic Features": [c for c in available_cols if c not in FEATURE_GROUPS["semantic"]],
        "w/o NLI Features": [c for c in available_cols if c not in FEATURE_GROUPS["nli"]],
        "w/o Entity Features": [c for c in available_cols if c not in FEATURE_GROUPS["entity"]],
        "w/o Sentence-Level Features": [c for c in available_cols if c not in FEATURE_GROUPS["sentence_level"]]
    }

    results = []
    benchmark_dict = {}

    y_train = train_feat["label"].values.astype(int)
    y_test = test_feat["label"].values.astype(int)

    base_f1 = None
    base_acc = None

    for config_name, feat_subset in configurations.items():
        logger.info(f"--- Running Ablation: {config_name} ({len(feat_subset)} features) ---")

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_feat[feat_subset].values)
        X_test = scaler.transform(test_feat[feat_subset].values)

        clf = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        benchmark_dict[config_name] = metrics

        if config_name == "Full Proposed Model":
            base_f1 = metrics["f1_score"]
            base_acc = metrics["accuracy"]
            delta_f1 = 0.0
            delta_acc = 0.0
        else:
            delta_f1 = round(metrics["f1_score"] - base_f1, 4)
            delta_acc = round(metrics["accuracy"] - base_acc, 4)

        results.append({
            "configuration": config_name,
            "num_features": len(feat_subset),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "macro_f1": metrics["macro_f1"],
            "roc_auc": metrics["roc_auc"],
            "delta_f1": delta_f1,
            "delta_accuracy": delta_acc
        })

    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv("results/metrics/ablation_results.csv", index=False)
    save_metrics(benchmark_dict, "results/metrics/ablation_results.json")

    print("\n" + "=" * 90)
    print("FEATURE ABLATION STUDY RESULTS")
    print("=" * 90)
    print(ablation_df.to_string(index=False))
    print("=" * 90 + "\n")

    return ablation_df


if __name__ == "__main__":
    run_ablation()
