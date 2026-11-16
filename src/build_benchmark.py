"""Build data/benchmark/questions.json for Day 7.

* 24 answerable questions come unchanged from the Day 6 benchmark (18 dev / 6 test, split by
  page so no test question shares a gold page with dev). They are converted to the Day 7
  schema: gold = [{source_id, page, evidence}].
* 4 unanswerable, near-topic questions are added (2 dev / 2 test). For each one the absence
  was checked with keyword searches over ALL 599 chunks (not just top-k); the terms and hit
  counts are stored in `absence_check` and re-verified every time this script runs.

Run: python -m src.build_benchmark  [--day6 <path to Day 6 benchmark/questions.json>]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import config
from .evidence import norm

DEFAULT_DAY6 = (config.ROOT.parents[1] / "14thang11" /
                "K5-L3B-Day06-VuTuanDat-2A202603744-DataIngestionEngineering" / "benchmark" / "questions.json")

UNANSWERABLE = [
    {"id": "u01", "split": "dev",
     "question": "How long does a patent last in Vietnam, and which office grants it?",
     "absence_reason": "The book covers U.S. law only; Vietnam and its IP office are never mentioned. "
                       "A U.S. patent term answer would not answer the question.",
     "absence_terms": [r"vietnam", r"viet nam"]},
    {"id": "u02", "split": "dev",
     "question": "What fee does WIPO charge for an international trademark application under the Madrid Protocol?",
     "absence_reason": "The book only says a U.S. registration can be a basis for foreign registration "
                       "(p.157); WIPO, the Madrid system and its fees are not covered.",
     "absence_terms": [r"wipo", r"madrid", r"international registration", r"protocol"]},
    {"id": "u03", "split": "test",
     "question": "Can a work created entirely by artificial intelligence, with no human author, be registered for copyright?",
     "absence_reason": "No passage discusses AI or non-human/computer-generated works; copyright chapters "
                       "only discuss authorship in general.",
     "absence_terms": [r"artificial intelligence", r"\bai\b", r"non-human", r"computer-generated", r"human author"]},
    {"id": "u04", "split": "test",
     "question": "Under the UDRP, how can a trademark owner recover a domain name from a cybersquatter?",
     "absence_reason": "Domain names, cybersquatting and the UDRP are not covered in the trademark chapter.",
     "absence_terms": [r"udrp", r"domain name", r"cybersquat"]},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day6", type=Path, default=DEFAULT_DAY6)
    args = ap.parse_args()
    chunks = json.loads(config.CORPUS_JSON.read_text(encoding="utf-8"))
    day6 = json.loads(args.day6.read_text(encoding="utf-8"))["questions"]

    out = []
    for q in day6:
        gold = [{"source_id": q["gold_source_id"], "page": p, "evidence": q["evidence_quote"]}
                for p in q["gold_page"]]
        resolving = [c["id"] for c in chunks for g in gold
                     if c["source_id"] == g["source_id"] and c["page"] == g["page"]
                     and norm(g["evidence"]) in norm(c["text"])]
        if not resolving:
            raise ValueError(f"{q['id']}: gold evidence not found inside one chunk on the gold page")
        out.append({"id": q["id"], "split": q["split"], "type": q["type"], "answerable": True,
                    "expected_behavior": "search_and_answer", "question": q["question"],
                    "expected_answer": q["expected_answer"], "gold": gold, "gold_chunk_ids": resolving,
                    "filter": q.get("filter")})

    for u in UNANSWERABLE:
        counts = {t: sum(bool(re.search(t, c["text"].lower())) for c in chunks) for t in u["absence_terms"]}
        if any(counts.values()):
            raise ValueError(f"{u['id']}: absence terms found in corpus: {counts}")
        out.append({"id": u["id"], "split": u["split"], "type": "unanswerable", "answerable": False,
                    "expected_behavior": "search_and_abstain", "question": u["question"],
                    "absence_reason": u["absence_reason"],
                    "absence_check": {"method": "regex over all chunk texts (lower-cased)",
                                      "chunks_matching_term": counts}, "filter": None})

    meta = {
        "corpus": f"{config.DOCUMENT_ID}, {len(chunks)} chunks (Day 6 strategy C)",
        "split_rule": "Answerable questions keep the Day 6 split (by page, no shared gold page). "
                      "Unanswerable: 2 dev, 2 test.",
        "test_history": "Day 6 test was run once to report Day 6 ingestion; it was never used to tune "
                        "any Day 7 retrieval/agent setting.",
        "hit_definition": "chunk from gold source AND on a gold page AND contains the full evidence "
                          "quote after normalization (case, quotes, dashes, whitespace).",
        "selection_rule_fixed_before_test": (
            "Choose the config with the highest dev Context Evidence Rate; ties -> higher dev mean "
            "grounded-answer grade; ties -> more correct abstentions; ties -> lower cost/latency. "
            "Retrieval method for the agent tool: highest dev Evidence Hit@3, tie -> Hit@1, tie -> MRR."),
    }
    config.QUESTIONS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.QUESTIONS_JSON.write_text(json.dumps({"meta": meta, "questions": out}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    n = lambda s, a: sum(q["split"] == s and q["answerable"] == a for q in out)
    print(f"dev: {n('dev', True)} answerable + {n('dev', False)} unanswerable; "
          f"test: {n('test', True)} answerable + {n('test', False)} unanswerable")
    for u in out[-4:]:
        print(u["id"], u["absence_check"]["chunks_matching_term"])


if __name__ == "__main__":
    main()
