"""
Diagnostic Verification Script for Cases A, B, and C.
Evaluates the updated LightweightHallucinationDetector on canonical test inputs.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extraction import FeatureExtractor
from src.detector import LightweightHallucinationDetector

def main():
    extractor = FeatureExtractor()
    detector = LightweightHallucinationDetector.load("models/hallucination_detector.joblib")

    cases = [
        (
            "CASE A — Strongly supported",
            "Python was created by Guido van Rossum and first released in 1991.",
            "Python was created by Guido van Rossum and first released in 1991."
        ),
        (
            "CASE B — Clearly contradictory",
            "Python was created by Guido van Rossum and first released in 1991.",
            "Python was created by James Gosling and first released in 2005."
        ),
        (
            "CASE C — Partially supported",
            "Python was created by Guido van Rossum and first released in 1991.",
            "Python was created by Guido van Rossum and first released in 1991. It was originally designed by Microsoft."
        ),
        (
            "USER REPORTED CASE — Grounded Sentence (TF-IDF ~0.914, Semantic ~0.995)",
            "Python was created by Guido van Rossum and released in 1991 as an open-source programming language.",
            "Python was developed by Guido van Rossum and released in 1991 as an open source language."
        )
    ]

    print("\n" + "=" * 80)
    print("HALLUCINATION DETECTOR DIAGNOSTIC EXPERIMENT RESULTS")
    print("=" * 80)

    for name, context, answer in cases:
        extracted = extractor.extract_single(context, answer)
        f = extracted["features"]
        explanation = detector.explain(f, extracted["sentence_analysis"], extracted["entity_details"])

        print(f"\nTest Case:             {name}")
        print(f"Context:               \"{context}\"")
        print(f"Answer:                \"{answer}\"")
        print(f"TF-IDF:                {f['tfidf_similarity']:.4f}")
        print(f"Semantic similarity:   {f['semantic_similarity']:.4f}")
        print(f"NLI entailment:        {f['nli_entailment_prob']:.4f}")
        print(f"NLI contradiction:     {f['nli_contradiction_prob']:.4f}")
        print(f"Entity score (unsupp): {f['entity_unsupported_count']:.0f} (ratio: {f['entity_unsupported_ratio']:.4f})")
        print(f"Predicted probability: {explanation['hallucination_probability'] * 100:.2f}%")
        print(f"Grounding score:       {explanation['supported_probability'] * 100:.2f}%")
        print(f"Predicted class:       {explanation['prediction']}")
        print(f"Heuristic score:       {explanation['heuristic_weighted_score'] * 100:.2f}%")
        print("Feature Contributions:")
        for feat, contrib in explanation["feature_contributions"]:
            print(f"  - {feat:26s}: {contrib:+.4f}")
        print("Evidence Points:")
        for ev in explanation["evidence"]:
            print(f"  - {ev}")
        print("-" * 80)

if __name__ == "__main__":
    main()
