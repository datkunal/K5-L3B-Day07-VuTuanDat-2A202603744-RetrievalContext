"""Run the real agent for one or more configs on a split and save traces + automatic metrics.

Automatic metrics here are only those that need no judgement: tool use, number of searches,
Context Evidence Rate (evidence quote present in the final context), status counts, latency, tokens,
cost. Answer grade, citation support and abstention correctness are labelled by reading the traces
(reviews.json) and aggregated with starter/agent_metrics.py.

Run: python -m src.run_agent --split dev --configs baseline_dense hybrid ...
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter

from . import config
from .agent import Agent, AgentConfig, context_chunks_for_metric
from .evidence import is_hit
from .retrievers import build

CONFIGS = {
    # three retrieval methods + rerank inside the same agent, single search on the original question
    "baseline_dense": AgentConfig("baseline_dense", method="dense", top_k=3, max_searches=1),
    "bm25":           AgentConfig("bm25", method="bm25", top_k=3, max_searches=1),
    "hybrid":         AgentConfig("hybrid", method="hybrid", top_k=3, max_searches=1),
    "rerank":         AgentConfig("rerank", method="rerank", top_k=3, max_searches=1),
    # improvement 1: wider context (more conditions/exceptions reach the model)
    "rerank_k5":      AgentConfig("rerank_k5", method="rerank", top_k=5, max_searches=1, budget_chars=6000),
    # improvement 2: agentic multi-search (model rewrites the query) + neighbour chunks
    "rerank_multi":   AgentConfig("rerank_multi", method="rerank", top_k=3, max_searches=3,
                                  budget_chars=6000, neighbors=True),
    # improvement 3 (added after reading dev errors d15/d03/d16): the model never searched a second
    # time; force one rephrased search before an abstain is accepted
    "dense_retry":    AgentConfig("dense_retry", method="dense", top_k=3, max_searches=3,
                                  budget_chars=6000, retry_before_abstain=True),
    "bm25_retry":     AgentConfig("bm25_retry", method="bm25", top_k=3, max_searches=3,
                                  budget_chars=6000, retry_before_abstain=True),
    "rerank_retry":   AgentConfig("rerank_retry", method="rerank", top_k=3, max_searches=3,
                                  budget_chars=6000, retry_before_abstain=True),
}


def load_questions(split: str) -> list[dict]:
    qs = json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))["questions"]
    return [q for q in qs if q["split"] == split]


def summarize(traces: list[dict], questions: list[dict]) -> dict:
    qa = {q["id"]: q for q in questions}
    ok = [t for t in traces if t["status"] != "error"]
    ans = [t for t in ok if qa[t["query_id"]]["answerable"]]
    una = [t for t in ok if not qa[t["query_id"]]["answerable"]]
    cer = [t for t in ans if t["context_has_evidence"]]
    lat = [t["latency_s"] for t in ok]
    costs = [t["cost_usd"] for t in ok if t["cost_usd"] is not None]
    return {
        "N": len(traces), "errors": len(traces) - len(ok),
        "tool_use": f"{sum(t['n_searches'] > 0 for t in ok)}/{len(ok)}",
        "searches_distribution": dict(sorted(Counter(t["n_searches"] for t in ok).items())),
        "context_evidence_rate": f"{len(cer)}/{len(ans)}",
        "context_evidence_rate_pct": round(len(cer) / len(ans), 3) if ans else None,
        "context_missing": [t["query_id"] for t in ans if not t["context_has_evidence"]],
        "answered_answerable": f"{sum(t['status'] == 'answered' for t in ans)}/{len(ans)}",
        "false_abstain_ids": [t["query_id"] for t in ans if t["status"] == "abstain"],
        "abstained_unanswerable": f"{sum(t['status'] == 'abstain' for t in una)}/{len(una)}",
        "answered_with_citation": f"{sum(bool(t['citations']) for t in ok if t['status'] == 'answered')}/"
                                  f"{sum(t['status'] == 'answered' for t in ok)}",
        "context_chars_mean": round(statistics.mean(t["context_chars"] for t in ok)) if ok else None,
        "prompt_tokens_total": sum(t["usage"]["prompt_tokens"] for t in ok),
        "completion_tokens_total": sum(t["usage"]["completion_tokens"] for t in ok),
        "llm_calls_total": sum(t["usage"]["llm_calls"] for t in ok),
        "cost_usd_total": round(sum(costs), 5), "latency_s_median": statistics.median(lat) if lat else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = ap.parse_args()
    questions = load_questions(args.split)
    cache: dict = {}
    rows = {}
    for name in args.configs:
        cfg = CONFIGS[name]
        agent = Agent(cfg, build(cfg.method, cache))
        traces = []
        for q in questions:
            t = agent.run(q["question"], filters=q.get("filter"))
            t = {"query_id": q["id"], "split": q["split"], "answerable": q["answerable"],
                 "question": q["question"], **t}
            t["context_has_evidence"] = (any(is_hit(q, c) for c in context_chunks_for_metric(t))
                                         if q["answerable"] else None)
            traces.append(t)
            print(f"  {name} {q['id']} {t['status']:8s} searches={t['n_searches']} "
                  f"ctx_evid={t['context_has_evidence']} cites={t['citations']}")
        out = config.RESULTS / "agent" / name / args.split
        out.mkdir(parents=True, exist_ok=True)
        (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
        rows[name] = summarize(traces, questions)
        (out / "summary.json").write_text(json.dumps(rows[name], indent=1), encoding="utf-8")
        print(name, json.dumps(rows[name]))
    cases = [{"id": q["id"], "split": q["split"], "answerable": q["answerable"]} for q in questions]
    (config.RESULTS / "agent" / f"cases_{args.split}.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
