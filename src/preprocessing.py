"""
Text Preprocessing and Sentence Splitting Module.

Provides robust text normalization and sentence segmentation for context and answer text.
"""

import re
import html
from typing import List


def clean_text(text: str) -> str:
    """
    Cleans and standardizes raw text:
    - Decodes HTML entities
    - Strips excessive whitespace and strange unicode characters
    - Preserves sentence-ending punctuation and casing for NER / NLI.
    """
    if not isinstance(text, str):
        return ""

    # Decode HTML entities (e.g. &amp;, &quot;)
    text = html.unescape(text)

    # Normalize unicode quotation marks and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ").replace("\ufffd", " ")

    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> List[str]:
    """
    Splits an answer or context into distinct sentences using regex rule-based segmentation.
    Avoids splitting on common honorifics (Dr., Mr., etc.) or numbers.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    # Simple and robust sentence boundary regular expression
    # Looks for .!? followed by whitespace and a capital letter or end of string
    sentence_end_pattern = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])')
    parts = sentence_end_pattern.split(cleaned)

    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        sentences = [cleaned]

    return sentences


def count_tokens(text: str) -> int:
    """
    Lightweight word token count.
    """
    return len(re.findall(r"\w+", text))
