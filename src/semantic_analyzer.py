"""
Semantic Similarity Analyzer Module.

Computes dense semantic embeddings using sentence-transformers/all-MiniLM-L6-v2.
Calculates cosine similarity at document and sentence levels.
Supports automatic device fallback (CUDA GPU -> CPU).
"""

import logging
from typing import List, Dict, Any, Union
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

from src.preprocessing import split_sentences

logger = logging.getLogger(__name__)


class SemanticSimilarityAnalyzer:
    """
    Lightweight embedding generator and cosine similarity analyzer.
    Uses mean-pooled MiniLM-L6-v2 representations.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
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

        logger.info(f"Initializing SemanticAnalyzer with {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Applies mean pooling to token embeddings accounting for attention mask.
        """
        token_embeddings = model_output[0]  # First element contains hidden states
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes a list of texts into normalized embedding vectors (shape: [N, 384]).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return torch.empty((0, 384), device=self.device)

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**encoded)
                pooled = self._mean_pooling(out, encoded["attention_mask"])
                normalized = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized)

        return torch.cat(all_embeddings, dim=0)

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        """
        Calculates cosine similarity between two text strings in range [-1, 1].
        """
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        sim = F.cosine_similarity(emb_a, emb_b).item()
        return float(sim)

    def analyze_pair(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Computes overall semantic similarity and sentence-level similarity breakdown.
        """
        ctx_emb = self.encode(context)
        ans_emb = self.encode(answer)
        overall_sim = float(F.cosine_similarity(ctx_emb, ans_emb).item())

        sentences = split_sentences(answer)
        if not sentences:
            return {
                "semantic_similarity": overall_sim,
                "min_sentence_semantic_sim": overall_sim,
                "mean_sentence_semantic_sim": overall_sim,
                "sentence_semantic_scores": []
            }

        sent_embs = self.encode(sentences)
        # Cosine similarity between each sentence and the context
        sent_sims = F.cosine_similarity(ctx_emb.expand(len(sentences), -1), sent_embs).tolist()

        return {
            "semantic_similarity": overall_sim,
            "min_sentence_semantic_sim": float(min(sent_sims)),
            "mean_sentence_semantic_sim": float(sum(sent_sims) / len(sent_sims)),
            "sentence_semantic_scores": [
                {"sentence": s, "similarity": round(float(sim), 4)}
                for s, sim in zip(sentences, sent_sims)
            ]
        }

    def batch_analyze(self, contexts: List[str], answers: List[str]) -> List[Dict[str, float]]:
        """
        Efficient batch analysis for datasets.
        """
        ctx_embs = self.encode(contexts)
        ans_embs = self.encode(answers)
        sims = F.cosine_similarity(ctx_embs, ans_embs).cpu().tolist()

        results = []
        for i, sim in enumerate(sims):
            results.append({
                "semantic_similarity": float(sim),
            })
        return results
