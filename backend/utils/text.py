"""Small text helpers shared by the agents."""

import re


def normalize_question(question: str) -> str:
    """Lowercase, trim and collapse spaces, so near-identical questions share a cache entry."""
    return re.sub(r"\s+", " ", question.strip().lower())
