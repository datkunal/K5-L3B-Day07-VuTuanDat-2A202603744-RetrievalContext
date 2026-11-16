"""Export the Day 6 Chroma collection (strategy C) to data/corpus/chunks.json in the Day 7 schema.

The vectors stay in Chroma; this JSON is used by lexical retrieval, evidence checks and the
report. Run: python -m src.export_day6
"""
from __future__ import annotations

import json

import chromadb

from . import config


def main() -> None:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    col = client.get_collection(config.CHROMA_COLLECTION)
    got = col.get(include=["documents", "metadatas", "embeddings"])
    dims = {len(v) for v in got["embeddings"]}
    if dims != {config.EMBEDDING_DIMS}:
        raise RuntimeError(f"unexpected vector dims in Day 6 index: {dims}")
    chunks = []
    for cid, text, md in zip(got["ids"], got["documents"], got["metadatas"]):
        chunks.append({
            "id": cid,
            "text": text,
            "source_id": md["document_id"],
            "source": md["source"],
            "page": int(md["page"]),
            "metadata": {k: md[k] for k in ("chapter", "section", "section_title", "ip_type",
                                            "printed_page", "char_start", "char_end", "content_sha1")
                         if k in md},
        })
    chunks.sort(key=lambda c: (c["page"], c["metadata"].get("char_start", 0)))
    config.CORPUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.CORPUS_JSON.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    pages = {c["page"] for c in chunks}
    print(f"exported {len(chunks)} chunks, {len(pages)} pages, dims={dims} -> {config.CORPUS_JSON}")


if __name__ == "__main__":
    main()
