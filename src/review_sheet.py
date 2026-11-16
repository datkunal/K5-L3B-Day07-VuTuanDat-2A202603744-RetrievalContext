"""Make a human review sheet (Markdown) + a reviews.json skeleton for one config/split.

The sheet shows question, expected answer (gold is shown to the REVIEWER only, never to the agent),
the agent answer, and the full text of every cited passage, so each citation can be read and
checked. Labels go into reviews.json (answer_grade 0-2, citation_support per citation,
abstention_correct for unanswerable, context_grade 0-2, note) and are aggregated with
starter/agent_metrics.py. Existing labels in reviews.json are kept.

Run: python -m src.review_sheet --config rerank_multi --split test
"""
from __future__ import annotations

import argparse
import json

from . import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    args = ap.parse_args()
    d = config.RESULTS / "agent" / args.config / args.split
    traces = json.loads((d / "traces.json").read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))["questions"]}
    reviews_path = d / "reviews.json"
    old = {r["query_id"]: r for r in json.loads(reviews_path.read_text(encoding="utf-8"))} if reviews_path.exists() else {}
    lines = [f"# Review sheet: {args.config} / {args.split}", ""]
    reviews = []
    for t in traces:
        q = qs[t["query_id"]]
        ctx = {c["id"]: c for c in t["final_context"]}
        lines += [f"## {t['query_id']} ({q['type']}) — {t['status']}", "",
                  f"**Q:** {q['question']}", ""]
        if q["answerable"]:
            lines += [f"**Expected:** {q['expected_answer']}", "",
                      f"**Gold evidence (p.{q['gold'][0]['page']}):** {q['gold'][0]['evidence']}", ""]
        else:
            lines += [f"**Absence reason:** {q['absence_reason']}", ""]
        lines += [f"**Agent:** {t['answer']}", ""]
        if t.get("abstain_reason"):
            lines += [f"**Abstain reason:** {t['abstain_reason']}", ""]
        lines += ["**Searches:** " + "; ".join(f"`{e['query']}` ({e['query_kind']}) -> "
                                              + ", ".join(f"p{r['page']}" for r in e["results"])
                                              for e in t["events"] if e["event"] == "search"), "",
                  f"**Context evidence (auto):** {t['context_has_evidence']} · context ids: "
                  + ", ".join(ctx), ""]
        for cid in t["citations"]:
            c = ctx.get(cid)
            lines += [f"> **[{cid}] p.{c['page'] if c else '?'}**  ", "> " + (c["text"] if c else "NOT IN CONTEXT").replace("\n", " "), ""]
        r = old.get(t["query_id"], {"query_id": t["query_id"]})
        r.setdefault("citation_support", [None] * len(t["citations"]))
        if q["answerable"]:
            r.setdefault("answer_grade", 0 if t["status"] == "abstain" else None)
            r.setdefault("context_grade", None)
        else:
            r.setdefault("abstention_correct", None)
        r.setdefault("note", "")
        reviews.append(r)
    (d / "review_sheet.md").write_text("\n".join(lines), encoding="utf-8")
    reviews_path.write_text(json.dumps(reviews, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {d / 'review_sheet.md'} and {reviews_path}")


if __name__ == "__main__":
    main()
