"""
Error Analysis and Diagnostic Profiling Module.

Categorizes test set predictions into:
- True Positives (TP): Correctly flagged hallucinations
- True Negatives (TN): Correctly verified supported answers
- False Positives (FP): Supported answers incorrectly flagged as hallucinated
- False Negatives (FN): Hallucinated answers that escaped detection

Extracts feature values, reasons, and produces error_analysis.csv.
Analyzes failure patterns for research discussion and presentation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """
    Performs comprehensive diagnostic error analysis on model predictions.
    """

    def __init__(self, output_path: str = "results/predictions/error_analysis.csv"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def analyze_predictions(
        self,
        test_df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Builds a comprehensive diagnostic DataFrame and exports error_analysis.csv.
        """
        analysis_df = test_df.copy()
        analysis_df["ground_truth"] = np.array(y_true).astype(int)
        analysis_df["prediction"] = np.array(y_pred).astype(int)
        analysis_df["hallucination_probability"] = np.round(np.array(y_prob), 4)

        # Classify prediction categories
        categories = []
        for yt, yp in zip(analysis_df["ground_truth"], analysis_df["prediction"]):
            if yt == 1 and yp == 1:
                categories.append("True Positive (TP)")
            elif yt == 0 and yp == 0:
                categories.append("True Negative (TN)")
            elif yt == 0 and yp == 1:
                categories.append("False Positive (FP - False Alarm)")
            elif yt == 1 and yp == 0:
                categories.append("False Negative (FN - Missed Hallucination)")
        analysis_df["error_category"] = categories

        # Diagnostic explanations for why prediction occurred
        diagnostic_reasons = []
        for _, row in analysis_df.iterrows():
            reasons = []
            cat = row["error_category"]
            c_prob = row.get("nli_contradiction_prob", 0.0)
            s_sim = row.get("semantic_similarity", 0.0)
            e_prob = row.get("nli_entailment_prob", 0.0)
            ent_unsupp = row.get("entity_unsupported_count", 0.0)

            if "FP" in cat:
                reasons.append("Answer supported by reference but flagged due to:")
                if c_prob > 0.40:
                    reasons.append(f"Spurious NLI contradiction ({c_prob:.2f})")
                if s_sim < 0.45:
                    reasons.append(f"Low surface semantic overlap ({s_sim:.2f}) despite factual truth")
                if ent_unsupp > 0:
                    reasons.append(f"{int(ent_unsupp)} novel entity/entities present in answer")
            elif "FN" in cat:
                reasons.append("Answer is hallucinated but slipped through due to:")
                if s_sim > 0.65:
                    reasons.append(f"High deceptive surface similarity ({s_sim:.2f}) masking subtle factual swap")
                if c_prob < 0.20:
                    reasons.append(f"Low NLI contradiction ({c_prob:.2f}), model missed subtle discrepancy")
                if ent_unsupp == 0:
                    reasons.append("Entities matched reference text, but semantic relation was corrupted")
            elif "TP" in cat:
                reasons.append(f"Correctly flagged: high contradiction ({c_prob:.2f}) or low semantic similarity ({s_sim:.2f})")
            else:
                reasons.append(f"Correctly verified: strong semantic alignment ({s_sim:.2f}) and high entailment ({e_prob:.2f})")

            diagnostic_reasons.append(" | ".join(reasons))

        analysis_df["diagnostic_explanation"] = diagnostic_reasons

        # Save to CSV
        analysis_df.to_csv(self.output_path, index=False)
        logger.info(f"Exported error analysis dataset to {self.output_path}")

        # Log category distribution
        counts = analysis_df["error_category"].value_counts().to_dict()
        logger.info(f"Error Distribution: {counts}")
        return analysis_df

    def summarize_failure_modes(self, analysis_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Summarizes key quantitative differences between error buckets for research discussion.
        """
        summary = {}
        for cat in ["True Positive (TP)", "True Negative (TN)", "False Positive (FP - False Alarm)", "False Negative (FN - Missed Hallucination)"]:
            subset = analysis_df[analysis_df["error_category"] == cat]
            if len(subset) > 0:
                summary[cat] = {
                    "count": len(subset),
                    "percentage": round(len(subset) / len(analysis_df) * 100, 2),
                    "mean_contradiction_prob": round(float(subset["nli_contradiction_prob"].mean()), 4) if "nli_contradiction_prob" in subset else None,
                    "mean_semantic_similarity": round(float(subset["semantic_similarity"].mean()), 4) if "semantic_similarity" in subset else None,
                    "mean_entailment_prob": round(float(subset["nli_entailment_prob"].mean()), 4) if "nli_entailment_prob" in subset else None,
                    "mean_unsupported_entities": round(float(subset["entity_unsupported_count"].mean()), 4) if "entity_unsupported_count" in subset else None
                }
        return summary
