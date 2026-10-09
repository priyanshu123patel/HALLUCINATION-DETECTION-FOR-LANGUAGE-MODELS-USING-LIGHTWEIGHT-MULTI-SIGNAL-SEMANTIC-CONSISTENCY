"""
Natural Language Inference (NLI) Analyzer Module.

Uses a lightweight pretrained Cross-Encoder NLI model (cross-encoder/nli-deberta-v3-xsmall)
to evaluate premise-hypothesis relationships:
- Premise: Reference Context
- Hypothesis: Generated Answer (or answer sentence)

Extracts Entailment, Contradiction, and Neutral probabilities.
Supports automatic device fallback (CUDA GPU -> CPU) and batch processing.
"""

import logging
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocessing import split_sentences

logger = logging.getLogger(__name__)


class NLIAnalyzer:
    """
    Evaluates factual consistency via Natural Language Inference.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-xsmall",
        device: str = "auto",
        batch_size: int = 32,
        max_length: int = 256
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Initializing NLIAnalyzer with {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Dynamic label mapping resolution from model configuration
        id2label = self.model.config.id2label
        self.label_indices = {}
        for idx, lbl in id2label.items():
            norm_lbl = str(lbl).lower()
            if "contradiction" in norm_lbl:
                self.label_indices["contradiction"] = int(idx)
            elif "entailment" in norm_lbl:
                self.label_indices["entailment"] = int(idx)
            elif "neutral" in norm_lbl:
                self.label_indices["neutral"] = int(idx)

        logger.info(f"NLI label indices: {self.label_indices}")

    def predict_probs(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Runs batched NLI inference over list of (premise, hypothesis) pairs.
        Returns list of dicts with 'contradiction', 'entailment', 'neutral' probabilities.
        """
        if not pairs:
            return []

        results = []
        c_idx = self.label_indices.get("contradiction", 0)
        e_idx = self.label_indices.get("entailment", 1)
        n_idx = self.label_indices.get("neutral", 2)

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            inputs = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

            for p in probs:
                p_list = p.tolist()
                results.append({
                    "nli_contradiction_prob": float(p_list[c_idx]),
                    "nli_entailment_prob": float(p_list[e_idx]),
                    "nli_neutral_prob": float(p_list[n_idx])
                })

        return results

    def analyze_pair(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Analyzes a single (context, answer) pair and produces document and sentence-level signals.
        """
        doc_probs = self.predict_probs([(context, answer)])[0]

        sentences = split_sentences(answer)
        if not sentences:
            return {
                **doc_probs,
                "max_sentence_contradiction_prob": doc_probs["nli_contradiction_prob"],
                "min_sentence_entailment_prob": doc_probs["nli_entailment_prob"],
                "sentence_nli_scores": []
            }

        sent_pairs = [(context, s) for s in sentences]
        sent_probs = self.predict_probs(sent_pairs)

        max_contr = max(sp["nli_contradiction_prob"] for sp in sent_probs)
        min_entail = min(sp["nli_entailment_prob"] for sp in sent_probs)

        sentence_scores = []
        for s, sp in zip(sentences, sent_probs):
            sentence_scores.append({
                "sentence": s,
                "contradiction": round(sp["nli_contradiction_prob"], 4),
                "entailment": round(sp["nli_entailment_prob"], 4),
                "neutral": round(sp["nli_neutral_prob"], 4),
            })

        return {
            **doc_probs,
            "max_sentence_contradiction_prob": float(max_contr),
            "min_sentence_entailment_prob": float(min_entail),
            "sentence_nli_scores": sentence_scores
        }
