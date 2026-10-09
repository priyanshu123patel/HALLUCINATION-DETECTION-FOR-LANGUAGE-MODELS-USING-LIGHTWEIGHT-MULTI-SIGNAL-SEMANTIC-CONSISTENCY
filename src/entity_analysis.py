"""
Named Entity Consistency Analysis Module.

Uses spaCy (en_core_web_sm) to extract named entities from context and answer,
comparing entity sets to identify unsupported or fabricated entities.

ACADEMIC LIMITATION DOCUMENTATION:
Entity consistency operates solely as a supporting feature. An entity present in the answer
but absent from the reference context is NOT guaranteed to be hallucinated (e.g. valid coreference,
paraphrase, or background knowledge). Conversely, an answer containing only context entities
may still hallucinate relationships between them. Hence, entity consistency is strictly
used in combination with semantic similarity and contradiction signals.
"""

import logging
from typing import Dict, Set, List, Any
import spacy

logger = logging.getLogger(__name__)

# Default entity types of interest for factual consistency
RELEVANT_ENTITY_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "DATE", "CARDINAL", "NORP", "FAC", "PRODUCT", "EVENT"
}


class EntityConsistencyAnalyzer:
    """
    Extracts named entities from context and answer, and computes consistency metrics.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name, disable=["parser"])
        except Exception as e:
            logger.warning(f"Could not load spaCy model '{model_name}': {e}. Loading blank 'en'.")
            self.nlp = spacy.blank("en")

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extracts named entities from a given text string.
        Returns a list of dicts with 'text', 'label', and 'normalized'.
        """
        if not text or not isinstance(text, str):
            return []

        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in RELEVANT_ENTITY_TYPES and len(ent.text.strip()) > 1:
                entities.append({
                    "text": ent.text.strip(),
                    "label": ent.label_,
                    "normalized": ent.text.strip().lower()
                })
        return entities

    def analyze(self, context: str, answer: str) -> Dict[str, Any]:
        """
        Compares entities between context and answer.
        Returns dictionary of metrics and entity lists.
        """
        ctx_ents = self.extract_entities(context)
        ans_ents = self.extract_entities(answer)

        ctx_norm_set: Set[str] = {e["normalized"] for e in ctx_ents}
        ans_norm_set: Set[str] = {e["normalized"] for e in ans_ents}

        # Check unsupported entities: in answer, but neither exact match nor substring of context
        context_lower = context.lower()
        unsupported: List[str] = []
        matched: List[str] = []

        for e in ans_ents:
            norm = e["normalized"]
            # Check if exact normalized entity is in context_norm_set OR appears in context text
            if (norm in ctx_norm_set) or (norm in context_lower):
                matched.append(e["text"])
            else:
                unsupported.append(e["text"])

        # Deduplicate while preserving order
        unsupported = list(dict.fromkeys(unsupported))
        matched = list(dict.fromkeys(matched))

        total_ans_entities = len(ans_norm_set)
        unsupported_count = len(unsupported)
        matched_count = len(matched)

        # Ratio of unsupported entities in answer
        unsupported_ratio = (unsupported_count / total_ans_entities) if total_ans_entities > 0 else 0.0

        # Jaccard overlap
        union_size = len(ctx_norm_set | ans_norm_set)
        intersection_size = len(ctx_norm_set & ans_norm_set)
        jaccard_overlap = (intersection_size / union_size) if union_size > 0 else 0.0

        return {
            "entity_unsupported_count": float(unsupported_count),
            "entity_unsupported_ratio": float(unsupported_ratio),
            "entity_jaccard_overlap": float(jaccard_overlap),
            "entity_total_answer": float(total_ans_entities),
            "entity_matched_count": float(matched_count),
            "unsupported_entity_list": unsupported,
            "matched_entity_list": matched,
        }
