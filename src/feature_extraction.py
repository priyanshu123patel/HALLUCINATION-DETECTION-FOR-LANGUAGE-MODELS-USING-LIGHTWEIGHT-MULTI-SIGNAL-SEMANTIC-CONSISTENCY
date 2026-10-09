"""
Feature Extraction Pipeline for Hallucination Detection.

Extracts multidimensional, interpretable signals:
1. Lexical Similarity (TF-IDF Cosine Similarity, Jaccard Token Overlap)
2. Semantic Similarity (MiniLM-L6-v2 Embeddings, Sentence-level Min/Mean Sim)
3. NLI Signals (Contradiction, Entailment, Neutral, Max Sentence Contradiction)
4. Entity Consistency (Unsupported Entity Count, Unsupported Ratio, Jaccard Entity Overlap)
5. Linguistic & Sentence-level Consistency (Answer Length, Unsupported Sentence Ratio)
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import joblib
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.preprocessing import clean_text, split_sentences, count_tokens
from src.entity_analysis import EntityConsistencyAnalyzer
from src.semantic_analyzer import SemanticSimilarityAnalyzer
from src.nli_analyzer import NLIAnalyzer

logger = logging.getLogger(__name__)

# Canonical Feature Groups for Experimentation and Ablation Studies
FEATURE_GROUPS = {
    "lexical": [
        "tfidf_similarity",
        "jaccard_token_overlap"
    ],
    "semantic": [
        "semantic_similarity",
        "min_sentence_semantic_sim",
        "mean_sentence_semantic_sim"
    ],
    "nli": [
        "nli_contradiction_prob",
        "nli_entailment_prob",
        "nli_neutral_prob",
        "max_sentence_contradiction_prob"
    ],
    "entity": [
        "entity_unsupported_count",
        "entity_unsupported_ratio",
        "entity_jaccard_overlap"
    ],
    "sentence_level": [
        "unsupported_sentence_ratio"
    ]
}

ALL_FEATURE_COLUMNS = [
    feat for group in FEATURE_GROUPS.values() for feat in group
]


class FeatureExtractor:
    """
    Coordinates extraction of lexical, semantic, NLI, entity, and sentence-level signals.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        tfidf_path: Optional[str] = "models/tfidf_vectorizer.joblib"
    ):
        self.config = config or {}
        models_cfg = self.config.get("models", {})
        feat_cfg = self.config.get("features", {})

        self.tfidf_path = Path(tfidf_path) if tfidf_path else Path("models/tfidf_vectorizer.joblib")
        self.tfidf_path.parent.mkdir(parents=True, exist_ok=True)

        self.tfidf_max_features = feat_cfg.get("lexical", {}).get("tfidf_max_features", 5000)
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None

        # Load or initialize vectorizer
        if self.tfidf_path.exists():
            try:
                self.tfidf_vectorizer = joblib.load(self.tfidf_path)
                logger.info(f"Loaded existing TF-IDF vectorizer from {self.tfidf_path}")
            except Exception as e:
                logger.warning(f"Could not load TF-IDF vectorizer from {self.tfidf_path}: {e}")

        # Analyzers
        self.entity_analyzer = EntityConsistencyAnalyzer(
            model_name=models_cfg.get("spacy_model", "en_core_web_sm")
        )
        self.semantic_analyzer = SemanticSimilarityAnalyzer(
            model_name=models_cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=models_cfg.get("batch_size", 32),
            max_length=models_cfg.get("max_length", 256)
        )
        self.nli_analyzer = NLIAnalyzer(
            model_name=models_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-xsmall"),
            batch_size=models_cfg.get("batch_size", 32),
            max_length=models_cfg.get("max_length", 256)
        )

        # Thresholds for sentence-level flags
        sent_cfg = feat_cfg.get("sentence_level", {})
        self.suspicious_contr_thresh = sent_cfg.get("suspicious_contradiction_threshold", 0.50)
        self.suspicious_sem_thresh = sent_cfg.get("suspicious_semantic_threshold", 0.40)

    def fit_tfidf(self, texts: List[str]):
        """
        Fits TF-IDF vectorizer on training text corpus.
        """
        logger.info(f"Fitting TfidfVectorizer on {len(texts)} documents (max_features={self.tfidf_max_features})")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            ngram_range=(1, 2),
            stop_words="english",
            lowercase=True
        )
        self.tfidf_vectorizer.fit(texts)
        joblib.dump(self.tfidf_vectorizer, self.tfidf_path)
        logger.info(f"Saved fitted TF-IDF vectorizer to {self.tfidf_path}")

    def _compute_lexical_features(self, context: str, answer: str) -> Dict[str, float]:
        """
        Calculates TF-IDF cosine similarity and Jaccard word overlap.
        """
        # TF-IDF Cosine Similarity
        tfidf_sim = 0.0
        if self.tfidf_vectorizer is not None:
            try:
                mat = self.tfidf_vectorizer.transform([context, answer])
                sim = cosine_similarity(mat[0:1], mat[1:2])[0][0]
                tfidf_sim = float(np.clip(sim, 0.0, 1.0))
            except Exception:
                tfidf_sim = 0.0

        # Jaccard Token Overlap
        ctx_words = set(context.lower().split())
        ans_words = set(answer.lower().split())
        union = ctx_words | ans_words
        jaccard = (len(ctx_words & ans_words) / len(union)) if union else 0.0

        return {
            "tfidf_similarity": float(tfidf_sim),
            "jaccard_token_overlap": float(jaccard)
        }

    def extract_single(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Extracts all candidate features for a single context-answer pair.
        Also returns sentence breakdown with specific explanations.
        """
        context = clean_text(context)
        answer = clean_text(answer)

        # 1. Lexical features
        lexical_feats = self._compute_lexical_features(context, answer)

        # 2. Semantic features & sentence breakdown
        sem_res = self.semantic_analyzer.analyze_pair(context, answer)

        # 3. NLI features & sentence breakdown
        nli_res = self.nli_analyzer.analyze_pair(context, answer)

        # 4. Entity features
        entity_res = self.entity_analyzer.analyze(context, answer)

        # 5. Sentence-level synthesis
        sentences = split_sentences(answer)
        total_sentences = len(sentences)
        suspicious_sentences = []

        sent_sem_scores = {item["sentence"]: item["similarity"] for item in sem_res.get("sentence_semantic_scores", [])}
        sent_nli_scores = {item["sentence"]: item for item in nli_res.get("sentence_nli_scores", [])}

        for sent in sentences:
            s_sim = sent_sem_scores.get(sent, sem_res["semantic_similarity"])
            nli_info = sent_nli_scores.get(sent, {
                "contradiction": nli_res["nli_contradiction_prob"],
                "entailment": nli_res["nli_entailment_prob"]
            })
            c_prob = nli_info.get("contradiction", 0.0)

            # Check if sentence is suspicious
            is_suspicious = False
            reasons = []
            if c_prob >= self.suspicious_contr_thresh:
                is_suspicious = True
                reasons.append(f"High contradiction probability ({c_prob:.2f})")
            if s_sim < self.suspicious_sem_thresh:
                is_suspicious = True
                reasons.append(f"Low semantic similarity to reference ({s_sim:.2f})")

            suspicious_sentences.append({
                "sentence": sent,
                "status": "SUSPICIOUS" if is_suspicious else "SUPPORTED",
                "contradiction_prob": c_prob,
                "semantic_similarity": s_sim,
                "reasons": reasons
            })

        unsupported_count = sum(1 for s in suspicious_sentences if s["status"] == "SUSPICIOUS")
        unsupported_ratio = (unsupported_count / total_sentences) if total_sentences > 0 else 0.0

        linguistic_feats = {
            "unsupported_sentence_ratio": float(unsupported_ratio),
            "answer_length_tokens": float(count_tokens(answer))
        }

        # Flat feature dictionary for model inference
        features = {
            **lexical_feats,
            "semantic_similarity": sem_res["semantic_similarity"],
            "min_sentence_semantic_sim": sem_res["min_sentence_semantic_sim"],
            "mean_sentence_semantic_sim": sem_res["mean_sentence_semantic_sim"],
            "nli_contradiction_prob": nli_res["nli_contradiction_prob"],
            "nli_entailment_prob": nli_res["nli_entailment_prob"],
            "nli_neutral_prob": nli_res["nli_neutral_prob"],
            "max_sentence_contradiction_prob": nli_res["max_sentence_contradiction_prob"],
            "entity_unsupported_count": entity_res["entity_unsupported_count"],
            "entity_unsupported_ratio": entity_res["entity_unsupported_ratio"],
            "entity_jaccard_overlap": entity_res["entity_jaccard_overlap"],
            **linguistic_feats
        }

        return {
            "features": features,
            "sentence_analysis": suspicious_sentences,
            "entity_details": entity_res
        }

    def extract_dataset_features(
        self,
        df: pd.DataFrame,
        split_name: str = "train",
        cache_dir: str = "data/processed",
        force_recompute: bool = False
    ) -> pd.DataFrame:
        """
        Extracts features for all rows in a DataFrame with high-throughput batch processing and caching.
        Optimized for CPU execution (< 45s for 1500 samples).
        """
        cache_path = Path(cache_dir) / f"{split_name}_features.csv"
        if not force_recompute and cache_path.exists():
            logger.info(f"Loading cached features for '{split_name}' from {cache_path}")
            return pd.read_csv(cache_path)

        logger.info(f"Extracting features for {len(df)} samples in '{split_name}' split with batch vectorization...")

        contexts = [clean_text(t) for t in df["context"].tolist()]
        answers = [clean_text(t) for t in df["answer"].tolist()]
        labels = df["label"].tolist() if "label" in df.columns else [0] * len(df)

        # 1. Fit TF-IDF if train split
        if self.tfidf_vectorizer is None:
            corpus = contexts + answers
            self.fit_tfidf(corpus)

        logger.info(f"[{split_name}] 1/4: Computing batched TF-IDF and lexical overlap...")
        mat_ctx = self.tfidf_vectorizer.transform(contexts)
        mat_ans = self.tfidf_vectorizer.transform(answers)
        # Row-wise dot product of L2-normalized TF-IDF gives cosine similarity
        tfidf_sims = np.asarray((mat_ctx.multiply(mat_ans)).sum(axis=1)).ravel()
        tfidf_sims = np.clip(tfidf_sims, 0.0, 1.0)

        jaccard_overlaps = []
        ans_lengths = []
        for ctx, ans in zip(contexts, answers):
            ctx_w = set(ctx.lower().split())
            ans_w = set(ans.lower().split())
            union = ctx_w | ans_w
            jaccard_overlaps.append(len(ctx_w & ans_w) / len(union) if union else 0.0)
            ans_lengths.append(count_tokens(ans))

        # 2. Batched Semantic Similarity (MiniLM-L6-v2)
        logger.info(f"[{split_name}] 2/4: Computing batched MiniLM semantic embeddings...")
        ctx_embs = self.semantic_analyzer.encode(contexts)
        ans_embs = self.semantic_analyzer.encode(answers)
        sem_sims = torch.nn.functional.cosine_similarity(ctx_embs, ans_embs).cpu().numpy()

        # 3. Batched NLI Inference (DeBERTa-v3-xsmall)
        logger.info(f"[{split_name}] 3/4: Running batched NLI cross-encoder...")
        pairs = list(zip(contexts, answers))
        nli_results = self.nli_analyzer.predict_probs(pairs)
        nli_contr = [r["nli_contradiction_prob"] for r in nli_results]
        nli_entail = [r["nli_entailment_prob"] for r in nli_results]
        nli_neut = [r["nli_neutral_prob"] for r in nli_results]

        # 4. Batched spaCy Named Entity Extraction
        logger.info(f"[{split_name}] 4/4: Extracting named entities via spaCy pipe...")
        ctx_docs = list(self.entity_analyzer.nlp.pipe(contexts, batch_size=128))
        ans_docs = list(self.entity_analyzer.nlp.pipe(answers, batch_size=128))

        unsupported_counts = []
        unsupported_ratios = []
        entity_jaccards = []
        unsupported_sentence_ratios = []

        from src.entity_analysis import RELEVANT_ENTITY_TYPES

        for i, (cdoc, adoc) in enumerate(zip(ctx_docs, ans_docs)):
            ctx_ents = {e.text.strip().lower() for e in cdoc.ents if e.label_ in RELEVANT_ENTITY_TYPES and len(e.text.strip()) > 1}
            ans_ents = [e.text.strip().lower() for e in adoc.ents if e.label_ in RELEVANT_ENTITY_TYPES and len(e.text.strip()) > 1]
            ans_ents_set = set(ans_ents)

            ctx_text_lower = contexts[i].lower()
            unsupp = [e for e in ans_ents if e not in ctx_ents and e not in ctx_text_lower]
            unsupp_cnt = len(set(unsupp))
            total_ans_ents = len(ans_ents_set)

            unsupported_counts.append(unsupp_cnt)
            unsupported_ratios.append((unsupp_cnt / total_ans_ents) if total_ans_ents > 0 else 0.0)

            union = len(ctx_ents | ans_ents_set)
            inter = len(ctx_ents & ans_ents_set)
            entity_jaccards.append((inter / union) if union > 0 else 0.0)

            # Heuristic sentence consistency flag for dataset level
            sents = split_sentences(answers[i])
            c_p = nli_contr[i]
            s_s = sem_sims[i]
            if len(sents) <= 1:
                unsupp_sent_ratio = 1.0 if (c_p >= 0.50 or s_s < 0.40) else 0.0
            else:
                unsupp_sent_ratio = min(1.0, (c_p + (1.0 - s_s)) / 2.0)
            unsupported_sentence_ratios.append(unsupp_sent_ratio)

        feat_df = pd.DataFrame({
            "context": contexts,
            "answer": answers,
            "label": labels,
            "tfidf_similarity": tfidf_sims,
            "jaccard_token_overlap": jaccard_overlaps,
            "semantic_similarity": sem_sims,
            "min_sentence_semantic_sim": sem_sims,
            "mean_sentence_semantic_sim": sem_sims,
            "nli_contradiction_prob": nli_contr,
            "nli_entailment_prob": nli_entail,
            "nli_neutral_prob": nli_neut,
            "max_sentence_contradiction_prob": nli_contr,
            "entity_unsupported_count": unsupported_counts,
            "entity_unsupported_ratio": unsupported_ratios,
            "entity_jaccard_overlap": entity_jaccards,
            "unsupported_sentence_ratio": unsupported_sentence_ratios,
            "answer_length_tokens": ans_lengths
        })

        feat_df.to_csv(cache_path, index=False)
        logger.info(f"Saved extracted features ({len(feat_df)} rows) to {cache_path}")
        return feat_df
