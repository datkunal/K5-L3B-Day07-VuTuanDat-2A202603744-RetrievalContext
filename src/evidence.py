"""Evidence matching shared by all metrics (same definition as Day 6)."""
from __future__ import annotations

import re
import unicodedata
from typing import Any


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", t).strip().lower()


def is_hit(question: dict[str, Any], chunk: dict[str, Any]) -> bool:
    """Chunk is evidence when it is from a gold source+page and contains the full quote."""
    return any(chunk["source_id"] == g["source_id"] and int(chunk["page"]) == int(g["page"])
               and norm(g["evidence"]) in norm(chunk["text"]) for g in question.get("gold", []))


def context_has_evidence(question: dict[str, Any], context_chunks: list[dict[str, Any]]) -> bool:
    """Context Evidence Rate numerator: some chunk in the FINAL context is a full evidence hit.

    Chunks may be truncated by the context packer; we check the text actually sent to the model.
    """
    return any(is_hit(question, c) for c in context_chunks)
