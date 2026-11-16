"""Retrieval methods over the Day 6 corpus.

1. dense   : e5-base query embedding -> Day 6 Chroma collection (cosine HNSW), optional metadata filter
2. bm25    : Okapi BM25 over the same chunks (lower-case word tokens, English stop words removed)
3. hybrid  : Reciprocal Rank Fusion (k=60) of dense and BM25 candidate lists (ranks, not raw scores,
             are fused, so no score normalization is needed)
4. rerank  : hybrid candidates re-scored by a cross-encoder (ms-marco MiniLM)

Every result: id, text, source_id, source, page, metadata, score, rank, method (+ per-stage info).
Filters come only from the question's user-provided `filter` field, never from gold.
"""
from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any

from . import config

STOP = set("""a an and are as at be by can could did do does for from has have how if in into is it its
may might must of on or so such than that the their them then there these they this to under was
were what when where which who whom why will with would""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


@lru_cache(maxsize=1)
def load_chunks() -> list[dict[str, Any]]:
    return json.loads(config.CORPUS_JSON.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def chunk_by_id() -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in load_chunks()}


def _check_filter(filters: dict | None) -> dict | None:
    if not filters:
        return None
    bad = set(filters) - config.ALLOWED_FILTER_FIELDS
    if bad:
        raise ValueError(f"filter fields not allowed: {bad}")
    return filters


def _passes(chunk: dict, filters: dict | None) -> bool:
    return not filters or all(chunk["metadata"].get(k) == v for k, v in filters.items())


def _result(chunk: dict, score: float, rank: int, method: str, **extra) -> dict[str, Any]:
    return {"id": chunk["id"], "text": chunk["text"], "source_id": chunk["source_id"],
            "source": chunk["source"], "page": chunk["page"], "metadata": chunk["metadata"],
            "score": round(float(score), 6), "rank": rank, "method": method, **extra}


class DenseRetriever:
    name = "dense"

    def __init__(self) -> None:
        import chromadb
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.EMBEDDING_MODEL)
        dims = self.model.get_sentence_embedding_dimension()
        if dims != config.EMBEDDING_DIMS:
            raise RuntimeError(f"query embedding dims {dims} != index dims {config.EMBEDDING_DIMS}")
        self.col = chromadb.PersistentClient(path=str(config.CHROMA_DIR)).get_collection(config.CHROMA_COLLECTION)
        if self.col.count() == 0:
            raise RuntimeError("Day 6 Chroma collection is empty")

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filter(filters)
        vec = self.model.encode([config.QUERY_PREFIX + query], normalize_embeddings=True)[0].tolist()
        where = None
        if filters:
            clauses = [{k: v} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self.col.query(query_embeddings=[vec], n_results=top_k, where=where, include=["distances"])
        by_id = chunk_by_id()
        return [_result(by_id[cid], 1.0 - d, r, self.name, cosine_distance=round(d, 6))
                for r, (cid, d) in enumerate(zip(res["ids"][0], res["distances"][0]), start=1)]


class BM25Retriever:
    name = "bm25"

    def __init__(self) -> None:
        from rank_bm25 import BM25Okapi
        self.chunks = load_chunks()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        filters = _check_filter(filters)
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], self.chunks[i]["id"]))
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            if _passes(self.chunks[i], filters):
                out.append(_result(self.chunks[i], scores[i], len(out) + 1, self.name))
                if len(out) == top_k:
                    break
        return out


class HybridRetriever:
    """RRF: score(d) = sum over retrievers of 1 / (k + rank_d)."""
    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, candidates: int = 20, k: int = 60) -> None:
        self.dense, self.bm25, self.candidates, self.k = dense, bm25, candidates, k

    def fuse(self, query: str, filters: dict | None) -> list[dict]:
        lists = {"dense": self.dense.search(query, filters=filters, top_k=self.candidates),
                 "bm25": self.bm25.search(query, filters=filters, top_k=self.candidates)}
        fused: dict[str, dict] = {}
        for src, res in lists.items():
            for r in res:
                e = fused.setdefault(r["id"], {"chunk": r, "rrf": 0.0, "ranks": {}})
                e["rrf"] += 1.0 / (self.k + r["rank"])
                e["ranks"][src] = r["rank"]
        return sorted(fused.values(), key=lambda e: (-e["rrf"], e["chunk"]["id"]))

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        fused = self.fuse(query, filters)[:top_k]
        by_id = chunk_by_id()
        return [_result(by_id[e["chunk"]["id"]], e["rrf"], r, self.name, source_ranks=e["ranks"])
                for r, e in enumerate(fused, start=1)]


class RerankRetriever:
    """Hybrid candidates (top `candidates` after RRF) re-ordered by a cross-encoder."""
    name = "rerank"

    def __init__(self, hybrid: HybridRetriever, candidates: int = 20) -> None:
        from sentence_transformers import CrossEncoder
        self.hybrid, self.candidates = hybrid, candidates
        self.ce = CrossEncoder(config.RERANK_MODEL)

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        fused = self.hybrid.fuse(query, filters)[: self.candidates]
        if not fused:
            return []
        by_id = chunk_by_id()
        scores = self.ce.predict([(query, by_id[e["chunk"]["id"]]["text"]) for e in fused])
        order = sorted(range(len(fused)), key=lambda i: (-scores[i], fused[i]["chunk"]["id"]))[:top_k]
        return [_result(by_id[fused[i]["chunk"]["id"]], scores[i], r, self.name,
                        rrf_rank=i + 1, source_ranks=fused[i]["ranks"])
                for r, i in enumerate(order, start=1)]


def build(method: str, cache: dict | None = None):
    """Factory sharing heavy models between methods."""
    cache = {} if cache is None else cache
    if "bm25" not in cache:
        cache["bm25"] = BM25Retriever()
    if method == "bm25":
        return cache["bm25"]
    if "dense" not in cache:
        cache["dense"] = DenseRetriever()
    if method == "dense":
        return cache["dense"]
    if "hybrid" not in cache:
        cache["hybrid"] = HybridRetriever(cache["dense"], cache["bm25"])
    if method == "hybrid":
        return cache["hybrid"]
    if method == "rerank":
        if "rerank" not in cache:
            cache["rerank"] = RerankRetriever(cache["hybrid"])
        return cache["rerank"]
    raise ValueError(f"unknown method {method}")


def timed_search(retriever, query: str, **kw) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    res = retriever.search(query, **kw)
    return res, round((time.perf_counter() - t0) * 1000, 1)
