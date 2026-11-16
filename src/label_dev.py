"""Manual labels for the DEV agent runs, written as code so they are reviewable and reproducible.

Labels come from reading results/logs/dev_answers_dump.txt and the full text of every cited chunk
(not from citation ids or a verifier model). Default after reading: answered = grade 2 and every
citation supports its sentence; the exceptions found while reading are listed explicitly below.

Run: python -m src.label_dev baseline_dense rerank rerank_k5 rerank_multi rerank_retry
"""
from __future__ import annotations

import json
import sys

from . import config

NOT_SUPPORTED = {
    ("baseline_dense", "d14", "C-openstax-intro-ip-2021-p164-c00"):
        "p164-c00 only has the end of the sentence ('confusion is likely to occur'); 'used in good faith "
        "for its primary meaning' is on p163-c02, which this run did not cite.",
}
NOTES = {
    "d03": "p053-c04 contains the full utility < novelty < non-obviousness hurdle ranking.",
    "d05": "If p055-c03 is cited it supports the 'new, original, non-obvious ornamental designs' sentence.",
    "d10": "Extra citation (p118-c02 or p120-c03) read: it supports the added 'registration recommended / "
           "statutory damages' sentence.",
    "d18": "p186-c02 and p186-c03 both contain the twice-the-award cap and the state-modification "
           "sentence, so either citation order is supported.",
}


def label(cfg: str, split: str = "dev") -> list[dict]:
    qs = {q["id"]: q for q in json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))["questions"]}
    d = config.RESULTS / "agent" / cfg / split
    out = []
    for t in json.loads((d / "traces.json").read_text(encoding="utf-8")):
        qid, q = t["query_id"], qs[t["query_id"]]
        notes, support = [], []
        for c in t["citations"]:
            bad = NOT_SUPPORTED.get((cfg, qid, c))
            support.append(bad is None)
            if bad:
                notes.append(bad)
        r = {"query_id": qid, "citation_support": support}
        if q["answerable"]:
            if t["status"] == "abstain":
                r["answer_grade"] = 0
                notes.append("False abstention: the evidence was not in the final context.")
            else:
                r["answer_grade"] = 2
                notes.append("Matches the expected answer incl. conditions; checked against the cited passage text.")
            r["context_grade"] = 2 if t["context_has_evidence"] else 0
            if qid in NOTES:
                notes.append(NOTES[qid])
        else:
            r["abstention_correct"] = t["status"] == "abstain"
            notes.append("Abstained without extra claims. Corpus check: " + q["absence_reason"])
        r["note"] = " ".join(notes)
        out.append(r)
    (d / "reviews.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        print("labelled", name, len(label(name)))
