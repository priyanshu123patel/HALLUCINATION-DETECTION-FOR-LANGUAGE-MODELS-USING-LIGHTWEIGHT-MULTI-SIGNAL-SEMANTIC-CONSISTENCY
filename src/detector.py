"""
Lightweight Multi-Signal Hallucination Detector.

Combines lexical similarity, dense semantic embeddings, NLI signals,
named entity consistency, and sentence-level signals using a calibrated,
monotonically consistent Logistic Regression model.

Key architectural properties:
1. Directional Semantic Monotonicity:
   - Contradiction and unsupported entities strictly increase hallucination risk.
   - Entailment, semantic similarity, and lexical overlap strictly reduce hallucination risk.
2. Invariant to Surface Length:
   - Spurious dataset length artifacts (e.g. HotpotQA 2-word answer shortcuts) are excluded.
3. Bounded [0, 1] Risk Representation:
   - Prevents unconstrained standard scaler explosions (no +11 sigma covariate shift).
4. Continuous & Calibrated Probabilities:
   - Output spans the full continuous [0, 1] interval without hard thresholds or saturation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

# Canonical directional risk features (all naturally bounded in [0.0, 1.0])
RISK_FEATURE_NAMES = [
    "contradiction_risk",
    "entity_unsupported_risk",
    "sentence_unsupported_risk",
    "non_entailment_risk",
    "semantic_distance_risk",
    "lexical_distance_risk"
]

FEATURE_DESCRIPTIONS = {
    "contradiction_risk": "Factual contradiction penalty (NLI premise-hypothesis conflict)",
    "entity_unsupported_risk": "Unsupported named entities penalty (novel entities in answer)",
    "sentence_unsupported_risk": "Ungrounded sentence ratio penalty (claim-level verification)",
    "non_entailment_risk": "Non-entailment penalty (1 - P(entailment))",
    "semantic_distance_risk": "Semantic divergence penalty (1 - MiniLM cosine similarity)",
    "lexical_distance_risk": "Lexical divergence penalty (1 - TF-IDF cosine similarity)"
}


def transform_to_risk_features(data: Union[pd.DataFrame, Dict[str, float], np.ndarray]) -> np.ndarray:
    """
    Transforms raw multi-signal features into the canonical 6-dimensional
    Hallucination Risk Vector in [0.0, 1.0].
    
    Signals:
    1. contradiction_risk: max(overall_contr, max_sentence_contr)
    2. entity_unsupported_risk: entity_unsupported_ratio
    3. sentence_unsupported_risk: unsupported_sentence_ratio
    4. non_entailment_risk: 1.0 - nli_entailment_prob
    5. semantic_distance_risk: 1.0 - semantic_similarity
    6. lexical_distance_risk: 1.0 - tfidf_similarity
    """
    if isinstance(data, dict):
        c = float(np.clip(data.get("nli_contradiction_prob", 0.0), 0.0, 1.0))
        e_un = float(np.clip(data.get("entity_unsupported_ratio", 0.0), 0.0, 1.0))
        s_un = float(np.clip(data.get("unsupported_sentence_ratio", 0.0), 0.0, 1.0))
        non_e = float(1.0 - np.clip(data.get("nli_entailment_prob", 0.0), 0.0, 1.0))
        s_sim = float(np.clip(data.get("semantic_similarity", 0.5), 0.0, 1.0))
        s_dist = 1.0 - s_sim
        t_sim = float(np.clip(data.get("tfidf_similarity", 0.0), 0.0, 1.0))
        t_dist = 1.0 - t_sim
        return np.array([[c, e_un, s_un, non_e, s_dist, t_dist]])

    elif isinstance(data, pd.DataFrame):
        df = data
        # Check if already risk features
        if all(col in df.columns for col in RISK_FEATURE_NAMES):
            return df[RISK_FEATURE_NAMES].values

        c = np.clip(df.get("nli_contradiction_prob", 0.0).fillna(0.0), 0.0, 1.0)
        e_un = np.clip(df.get("entity_unsupported_ratio", 0.0).fillna(0.0), 0.0, 1.0)
        s_un = np.clip(df.get("unsupported_sentence_ratio", 0.0).fillna(0.0), 0.0, 1.0)
        non_e = 1.0 - np.clip(df.get("nli_entailment_prob", 0.0).fillna(0.0), 0.0, 1.0)
        s_sim = np.clip(df.get("semantic_similarity", 0.5).fillna(0.5), 0.0, 1.0)
        s_dist = 1.0 - s_sim
        t_sim = np.clip(df.get("tfidf_similarity", 0.0).fillna(0.0), 0.0, 1.0)
        t_dist = 1.0 - t_sim
        return np.column_stack([c, e_un, s_un, non_e, s_dist, t_dist])

    else:
        arr = np.array(data)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr


class LightweightHallucinationDetector:
    """
    Primary proposed classifier for hallucination detection.
    
    Uses calibrated, monotonically-constrained Logistic Regression over
    normalized factual consistency and divergence signals.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        config: Optional[dict] = None,
        C: float = 1.0,
        class_weight: str = "balanced"
    ):
        self.config = config or {}
        self.feature_columns = feature_columns
        self.feature_names = RISK_FEATURE_NAMES
        self.is_fitted = False
        self.scaler = None  # Naturally bounded in [0, 1]; no uncalibrated scaler needed

        self.classifier = LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=1000,
            random_state=42
        )

        # Heuristic weighted scoring configuration (Method B baseline)
        scoring_cfg = self.config.get("weighted_scoring", {})
        self.w_contradiction = scoring_cfg.get("contradiction_weight", 0.35)
        self.w_semantic = scoring_cfg.get("semantic_similarity_weight", -0.25)
        self.w_entailment = scoring_cfg.get("entailment_weight", -0.20)
        self.w_entity = scoring_cfg.get("entity_unsupported_weight", 0.20)
        self.w_sentence = scoring_cfg.get("unsupported_sentence_weight", 0.15)
        self.weighted_threshold = scoring_cfg.get("threshold", 0.50)

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """
        Fits the detector using penalized monotonic logistic optimization.
        
        Learns optimal weights on benchmark data while enforcing directional
        semantic validity:
        - beta_k >= 0 for all risk features (contradiction, unsupported entities,
          non-entailment, semantic divergence, lexical divergence).
        - Regularized shrinkage towards semantic grounding priors.
        """
        R = transform_to_risk_features(X)
        y_vec = np.array(y).astype(int)

        logger.info(
            f"Fitting LightweightHallucinationDetector on {len(R)} samples with {R.shape[1]} risk features."
        )

        # Directional semantic prior weights
        # Prior encodes domain logic: contradiction and entity hallucinations are decisive signals,
        # while non-entailment, sentence ungroundedness, and semantic distance provide continuous gradients.
        w_prior = np.array([2.5, 4.0, 1.2, 0.6, 0.6, 0.3])
        b_prior = -2.8
        alpha = 0.005

        def loss_fn(params):
            b = params[0]
            w = params[1:]
            z = np.clip(b + np.dot(R, w), -30.0, 30.0)
            p_hat = 1.0 / (1.0 + np.exp(-z))
            eps = 1e-12
            bce = -np.mean(y_vec * np.log(p_hat + eps) + (1.0 - y_vec) * np.log(1.0 - p_hat + eps))
            reg = alpha * (np.sum((w - w_prior) ** 2) + (b - b_prior) ** 2)
            return bce + reg

        # Non-negative bounds ensure semantic consistency
        bounds = [(-6.0, 0.0)] + [(0.2, 8.0) for _ in range(6)]
        init_params = [b_prior] + list(w_prior)
        opt_res = minimize(loss_fn, init_params, bounds=bounds, method="L-BFGS-B")

        b_opt = float(opt_res.x[0])
        w_opt = opt_res.x[1:]

        # Configure Scikit-Learn LogisticRegression instance
        self.classifier.coef_ = np.array([w_opt])
        self.classifier.intercept_ = np.array([b_opt])
        self.classifier.classes_ = np.array([0, 1])
        self.feature_names = RISK_FEATURE_NAMES
        self.is_fitted = True

        logger.info(f"Fitted Intercept (b): {b_opt:+.4f}")
        logger.info("Fitted Risk Feature Weights (w):")
        for feat, val in zip(self.feature_names, w_opt):
            logger.info(f"  - {feat:26s}: {val:+.4f} ({FEATURE_DESCRIPTIONS.get(feat, '')})")

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, float]]) -> np.ndarray:
        """
        Returns predicted probabilities: [P(SUPPORTED), P(HALLUCINATED)].
        """
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted yet. Call fit() or load().")

        R = transform_to_risk_features(X)
        return self.classifier.predict_proba(R)

    def predict(self, X: Union[pd.DataFrame, np.ndarray, Dict[str, float]], threshold: float = 0.50) -> np.ndarray:
        """
        Returns binary predictions: 0 = SUPPORTED, 1 = HALLUCINATED.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def compute_weighted_score(self, features: Dict[str, float]) -> Tuple[float, int]:
        """
        Computes the interpretable heuristic weighted score (Method B baseline).
        Normalized score in [0.0, 1.0].
        """
        c_prob = features.get("nli_contradiction_prob", 0.0)
        s_sim = features.get("semantic_similarity", 0.5)
        e_prob = features.get("nli_entailment_prob", 0.5)
        ent_unsupp = features.get("entity_unsupported_ratio", 0.0)
        sent_unsupp = features.get("unsupported_sentence_ratio", 0.0)

        score = (
            abs(self.w_contradiction) * c_prob +
            abs(self.w_semantic) * (1.0 - np.clip(s_sim, 0.0, 1.0)) +
            abs(self.w_entailment) * (1.0 - np.clip(e_prob, 0.0, 1.0)) +
            abs(self.w_entity) * np.clip(ent_unsupp, 0.0, 1.0) +
            abs(self.w_sentence) * np.clip(sent_unsupp, 0.0, 1.0)
        )
        norm_score = float(np.clip(score, 0.0, 1.0))
        pred_label = 1 if norm_score >= self.weighted_threshold else 0
        return norm_score, pred_label

    def explain(
        self,
        features: Dict[str, float],
        sentence_analysis: Optional[List[Dict[str, Any]]] = None,
        entity_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates mathematically grounded explanations for a prediction.
        Computes exact logit and linear feature contributions:
        logit = beta_0 + sum_k (beta_k * risk_k)
        """
        R = transform_to_risk_features(features)[0]
        probs = self.predict_proba(features)[0]
        hallu_prob = float(probs[1])
        pred_class = "HALLUCINATED" if hallu_prob >= 0.50 else "SUPPORTED"

        # Calculate exact linear contribution of each risk signal towards hallucination logit
        intercept = float(self.classifier.intercept_[0])
        contributions = {}
        for idx, col in enumerate(self.feature_names):
            coef = float(self.classifier.coef_[0][idx])
            risk_val = float(R[idx])
            contributions[col] = float(coef * risk_val)

        sorted_contribs = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

        # Generate human-readable academic evidence points
        evidence_points = []

        # Contradiction check
        c_prob = features.get("nli_contradiction_prob", 0.0)
        if c_prob >= 0.40:
            evidence_points.append(
                f"Contradiction probability is high ({c_prob:.2f}), strongly indicating factual conflict."
            )
        elif c_prob <= 0.05:
            evidence_points.append(
                f"Contradiction probability is minimal ({c_prob:.4f}), indicating no direct factual conflict."
            )

        # Semantic similarity check
        sem_sim = features.get("semantic_similarity", 0.0)
        if sem_sim >= 0.85:
            evidence_points.append(
                f"High semantic alignment with reference context ({sem_sim:.2f})."
            )
        elif sem_sim < 0.45:
            evidence_points.append(
                f"Low semantic similarity to reference context ({sem_sim:.2f})."
            )

        # Entailment check
        ent_prob = features.get("nli_entailment_prob", 0.0)
        if ent_prob >= 0.60:
            evidence_points.append(
                f"High entailment probability ({ent_prob:.2f}), candidate is directly inferred from context."
            )
        elif ent_prob < 0.15:
            evidence_points.append(
                f"Low entailment probability ({ent_prob:.2f}), context fails to logically entail the statement."
            )

        # Entity check
        if entity_details:
            unsupp_list = entity_details.get("unsupported_entity_list", [])
            if unsupp_list:
                evidence_points.append(
                    f"{len(unsupp_list)} unsupported entity/entities detected in answer: {unsupp_list}."
                )
            else:
                evidence_points.append(
                    "All extracted named entities in the answer match the reference context."
                )

        # Sentence-level check
        if sentence_analysis:
            suspicious_count = sum(1 for s in sentence_analysis if s.get("status") == "SUSPICIOUS")
            if suspicious_count > 0:
                evidence_points.append(
                    f"{suspicious_count}/{len(sentence_analysis)} sentence(s) flagged as suspicious."
                )
            else:
                evidence_points.append("All individual sentences verified as grounded.")

        w_score, w_pred = self.compute_weighted_score(features)

        return {
            "prediction": pred_class,
            "hallucination_probability": round(hallu_prob, 4),
            "supported_probability": round(1.0 - hallu_prob, 4),
            "heuristic_weighted_score": round(w_score, 4),
            "heuristic_prediction": "HALLUCINATED" if w_pred == 1 else "SUPPORTED",
            "evidence": evidence_points,
            "feature_contributions": sorted_contribs,
            "sentence_analysis": sentence_analysis or []
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Returns DataFrame of feature names, coefficients, and descriptions.
        """
        if not self.is_fitted:
            raise RuntimeError("Detector is not fitted yet.")

        coefs = self.classifier.coef_[0]
        df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefs,
            "weight": np.abs(coefs),
            "description": [FEATURE_DESCRIPTIONS.get(f, "") for f in self.feature_names],
            "direction": ["Hallucinating (+)" for _ in coefs]
        }).sort_values("weight", ascending=False).reset_index(drop=True)
        return df

    def save(self, filepath: str = "models/hallucination_detector.joblib"):
        """
        Serializes detector pipeline to disk.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": None,
            "classifier": self.classifier,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "config": self.config
        }, path)
        logger.info(f"Detector successfully saved to {path}")

    @classmethod
    def load(cls, filepath: str = "models/hallucination_detector.joblib") -> "LightweightHallucinationDetector":
        """
        Loads serialized detector pipeline.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Detector checkpoint not found at: {path}")

        checkpoint = joblib.load(path)
        detector = cls(
            feature_columns=checkpoint.get("feature_names", RISK_FEATURE_NAMES),
            config=checkpoint.get("config", {})
        )
        detector.scaler = checkpoint.get("scaler", None)
        detector.classifier = checkpoint["classifier"]
        detector.feature_names = checkpoint.get("feature_names", RISK_FEATURE_NAMES)
        detector.is_fitted = checkpoint.get("is_fitted", True)
        logger.info(f"Detector successfully loaded from {path}")
        return detector
