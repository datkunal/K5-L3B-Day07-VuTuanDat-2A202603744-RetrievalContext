"""Manual labels for the TEST runs (baseline_dense, rerank), one explicit entry per question.

Every answer and every cited chunk was read in full (results/logs/test_answers_dump.txt). Labels
were assigned after the config was frozen (results/selection.json); nothing was changed afterwards.

Run: python -m src.label_test
"""
from __future__ import annotations

import json

from . import config

LABELS = {
    "baseline_dense": {
        "t01": (2, 2, "Correct: CC licences cannot limit fair use; p132-c01 contains the exact sentence."),
        "t02": (2, 2, "Correct: brand-reputation reason; second sentence (injunction prohibits continued sales) "
                      "also stated in p162-c01."),
        "t03": (2, 2, "Gold quote (p185-c00) NOT in context, but valid alternative evidence was: p185-c01 "
                      "('duration ... time required for independent development', head-start = proper means) "
                      "and p184-c02 (injunction terminates when the secret ceases to exist). Answer covers both "
                      "parts; all 3 citations support their sentences. Automatic CER counts this as a miss "
                      "because the gold list has only one evidence passage."),
        "t04": (2, 2, "Correct: secondary meaning, Sharp/Windows; p152-c04 supports the 'established over time "
                      "through sales and advertising' sentence."),
        "t05": (2, 2, "Correct: doctrine of equivalents, function-way-result test, from p067-c01."),
        "t06": (2, 2, "Correct: 250 words of a poem, p122-c05."),
        "u03": (True, None, "Abstained; context was general copyright/authorship passages with nothing on AI. "
                            "No extra claim made."),
        "u04": (True, None, "Abstained; context was trademark registration passages, nothing on UDRP/domains."),
    },
    "rerank": {
        "t01": (2, 2, "Correct; second sentence ('broad dissemination') is supported by p132-c01."),
        "t02": (2, 2, "Correct: brand-reputation reason, p162-c01."),
        "t03": (1, 2, "Partly correct: duration = time for independent development (p185-c01) and courts may rely "
                      "on experts (p185-c00) are supported, but the answer omits that the injunction ends when "
                      "the trade secret ceases to exist, although p185-c00 with that sentence was in context."),
        "t04": (2, 2, "Correct: secondary meaning, Sharp/Windows, p153-c03."),
        "t05": (2, 2, "Correct: doctrine of equivalents, p067-c01."),
        "t06": (2, 2, "Correct: 250 words of a poem, p122-c05."),
        "u03": (True, None, "Abstained; retrieved copyright passages say nothing about AI-generated works."),
        "u04": (True, None, "Abstained; nothing on UDRP or domain names in context."),
    },
}


def main() -> None:
    for cfg, labels in LABELS.items():
        d = config.RESULTS / "agent" / cfg / "test"
        traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
        out = []
        for t in traces:
            a, ctx, note = labels[t["query_id"]]
            r = {"query_id": t["query_id"], "citation_support": [True] * len(t["citations"]), "note": note}
            if t["answerable"]:
                r["answer_grade"], r["context_grade"] = a, ctx
            else:
                r["abstention_correct"] = a and t["status"] == "abstain"
            out.append(r)
        (d / "reviews.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print("labelled", cfg, len(out))


if __name__ == "__main__":
    main()
