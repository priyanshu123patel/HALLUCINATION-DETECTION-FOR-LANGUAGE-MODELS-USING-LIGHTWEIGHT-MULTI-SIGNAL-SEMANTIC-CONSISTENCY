"""
Smoke Test Script for Hallucination Detection Pipeline.

Runs a fast end-to-end verification across:
1. Data loading
2. Feature extraction (Lexical, Semantic, NLI, Entity)
3. Model fitting & prediction
4. Explanation generation
5. Evaluation metrics calculation
6. Visualization generation
"""

import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np

from src.preprocessing import clean_text, split_sentences
from src.entity_analysis import EntityConsistencyAnalyzer
from src.semantic_analyzer import SemanticSimilarityAnalyzer
from src.nli_analyzer import NLIAnalyzer
from src.feature_extraction import FeatureExtractor
from src.detector import LightweightHallucinationDetector
from src.evaluation import compute_metrics
from src import visualization as viz

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_smoke_test():
    logger.info(">>> STEP 1: Testing Preprocessing & Sentence Splitting...")
    text = "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia. First for Women was started later."
    cleaned = clean_text(text)
    sents = split_sentences(cleaned)
    assert len(sents) == 2, f"Expected 2 sentences, got {len(sents)}"
    logger.info(f"Splits verified: {sents}")

    logger.info(">>> STEP 2: Testing spaCy Entity Consistency Analyzer...")
    entity_analyzer = EntityConsistencyAnalyzer("en_core_web_sm")
    context = "Arthur's Magazine was published in Philadelphia in the 19th century."
    answer_supp = "Arthur's Magazine was published in Philadelphia."
    answer_hallu = "Arthur's Magazine was published in Tokyo by Sony Corporation."

    res_supp = entity_analyzer.analyze(context, answer_supp)
    res_hallu = entity_analyzer.analyze(context, answer_hallu)
    logger.info(f"Supported answer unsupported entities: {res_supp['unsupported_entity_list']}")
    logger.info(f"Hallucinated answer unsupported entities: {res_hallu['unsupported_entity_list']}")
    assert len(res_hallu["unsupported_entity_list"]) > 0, "Failed to detect unsupported entities in hallucinated answer!"

    logger.info(">>> STEP 3: Testing Semantic Similarity Analyzer...")
    sem_analyzer = SemanticSimilarityAnalyzer(max_length=128)
    sim_supp = sem_analyzer.compute_similarity(context, answer_supp)
    sim_hallu = sem_analyzer.compute_similarity(context, "Astronauts landed on the surface of Mars in 1969.")
    logger.info(f"Semantic sim (supported): {sim_supp:.4f} | Semantic sim (unrelated): {sim_hallu:.4f}")
    assert sim_supp > sim_hallu, "Semantic similarity failed to score supported text higher than unrelated text!"

    logger.info(">>> STEP 4: Testing NLI Analyzer...")
    nli_analyzer = NLIAnalyzer(max_length=128)
    nli_supp = nli_analyzer.predict_probs([(context, answer_supp)])[0]
    nli_contr = nli_analyzer.predict_probs([(context, "Arthur's Magazine was never published in Philadelphia.")])[0]
    logger.info(f"NLI Supported -> Entailment: {nli_supp['nli_entailment_prob']:.4f}, Contradiction: {nli_supp['nli_contradiction_prob']:.4f}")
    logger.info(f"NLI Contradiction -> Entailment: {nli_contr['nli_entailment_prob']:.4f}, Contradiction: {nli_contr['nli_contradiction_prob']:.4f}")
    assert nli_contr["nli_contradiction_prob"] > nli_supp["nli_contradiction_prob"], "NLI failed to detect high contradiction!"

    logger.info(">>> STEP 5: Testing Feature Extractor single-instance and explanations...")
    extractor = FeatureExtractor()
    sample_res = extractor.extract_single(context, answer_hallu)
    features = sample_res["features"]
    assert "tfidf_similarity" in features
    assert "semantic_similarity" in features
    assert "nli_contradiction_prob" in features
    assert "entity_unsupported_count" in features

    logger.info(">>> STEP 6: Testing Detector Fit & Explanation...")
    # Synthetic mini-dataset for smoke testing
    mock_df = pd.DataFrame([
        {**features, "label": 1},
        {**extractor.extract_single(context, answer_supp)["features"], "label": 0},
        {**features, "label": 1},
        {**extractor.extract_single(context, answer_supp)["features"], "label": 0}
    ])
    feat_cols = list(features.keys())
    detector = LightweightHallucinationDetector(feature_columns=feat_cols)
    detector.fit(mock_df[feat_cols], mock_df["label"])

    explanation = detector.explain(features, sample_res["sentence_analysis"], sample_res["entity_details"])
    logger.info(f"Detector explanation verdict: {explanation['prediction']} (P={explanation['hallucination_probability']})")
    assert "evidence" in explanation
    assert len(explanation["evidence"]) > 0

    logger.info(">>> STEP 7: Testing Evaluation Metrics & Visualizations...")
    m = compute_metrics([1, 0, 1, 0], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert m["accuracy"] == 1.0

    viz.plot_sentence_level_example(sample_res["sentence_analysis"], "results/figures/smoke_test_sentence.png")
    assert Path("results/figures/smoke_test_sentence.png").exists()

    logger.info("======================================================================")
    logger.info("ALL SMOKE TESTS PASSED SUCCESSFULLY! PIPELINE IS VERIFIED!")
    logger.info("======================================================================")

if __name__ == "__main__":
    run_smoke_test()
