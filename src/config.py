"""Day 7 settings: Day 6 connection, models, prices and default experiment knobs."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS_JSON = DATA / "corpus" / "chunks.json"
QUESTIONS_JSON = DATA / "benchmark" / "questions.json"
RESULTS = ROOT / "results"

try:  # .env holds OPENAI_API_KEY / OPENAI_MODEL; never committed
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

# ---- Day 6 vector store (reused as-is, not re-indexed) -------------------------------
# Day 6 chose strategy C (paragraph chunks <=1000 chars + ip_type filter). Chroma 1.5.9
# cannot reopen an HNSW index under a non-ASCII path, so Day 6 stored it in LOCALAPPDATA.
def _chroma_dir() -> Path:
    env = os.getenv("DAY06_CHROMA_DIR")
    if env:
        return Path(env)
    base = os.getenv("LOCALAPPDATA")
    return Path(base) / "day06_chroma" if base else Path.home() / ".day06_chroma"


CHROMA_DIR = _chroma_dir()
CHROMA_COLLECTION = "day06_strategy_C"
DOCUMENT_ID = "openstax-intro-ip-2021"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"   # same as Day 6 index, 768 dims
EMBEDDING_DIMS = 768
QUERY_PREFIX = "query: "

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---- Agent model -----------------------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# USD per 1M tokens (input, output). Edit if the price list changes; used only for reporting.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-mini": (0.40, 1.60), "gpt-4o": (2.50, 10.00)}

ALLOWED_FILTER_FIELDS = {"ip_type", "chapter", "section"}
