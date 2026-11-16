"""Retrieval-only benchmark: Hit@1, Evidence Hit@3, MRR@10, latency, on the ORIGINAL question text.

Unanswerable questions are excluded from the denominators (BENCHMARK.md). For them we only log
the top score, to see whether a score threshold could signal "no evidence".

Run: python -m src.eval_retrieval --split dev [--methods dense bm25 hybrid rerank] [--no-filter]
"""
from __future__ import annotations

import argparse
import json
import statistics

from . import config
from .evidence import is_hit
from .retrievers import build, timed_search

K = 10


def evaluate(method: str, questions: list[dict], cache: dict, use_filter: bool) -> tuple[dict, list[dict]]:
    r = build(method, cache)
    traces, lat = [], []
    for q in questions:
        flt = q.get("filter") if use_filter else None
        res, ms = timed_search(r, q["question"], filters=flt, top_k=K)
        lat.append(ms)
        flags = [is_hit(q, c) for c in res] if q["answerable"] else []
        first = next((i + 1 for i, f in enumerate(flags) if f), None)
        traces.append({"method": method, "use_filter": use_filter, "id": q["id"], "split": q["split"],
                       "answerable": q["answerable"], "question": q["question"], "filter": flt,
                       "hit_at_1": bool(flags[:1] and flags[0]), "hit_at_3": any(flags[:3]),
                       "first_hit_rank": first, "latency_ms": ms,
                       "top": [{"rank": c["rank"], "id": c["id"], "page": c["page"], "score": c["score"],
                                "is_evidence": f if q["answerable"] else None,
                                **({"source_ranks": c["source_ranks"]} if "source_ranks" in c else {})}
                               for c, f in zip(res, flags or [None] * len(res))]})
    ans = [t for t in traces if t["answerable"]]
    n = len(ans)
    summary = {"method": method, "use_filter": use_filter, "N": n,
               "hit1": sum(t["hit_at_1"] for t in ans), "hit3": sum(t["hit_at_3"] for t in ans),
               "mrr10": round(sum(1 / t["first_hit_rank"] for t in ans if t["first_hit_rank"]) / n, 3),
               "miss_at_10": [t["id"] for t in ans if not t["first_hit_rank"]],
               "miss_at_3": [t["id"] for t in ans if not t["hit_at_3"]],
               "latency_ms_median": statistics.median(lat),
               "unanswerable_top1_score": {t["id"]: t["top"][0]["score"] if t["top"] else None
                                           for t in traces if not t["answerable"]},
               "answerable_top1_score_median": statistics.median(t["top"][0]["score"] for t in ans if t["top"])}
    summary["hit1_rate"] = round(summary["hit1"] / n, 3)
    summary["hit3_rate"] = round(summary["hit3"] / n, 3)
    return summary, traces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid", "rerank"])
    ap.add_argument("--no-filter", action="store_true", help="also run without the user-provided filter")
    args = ap.parse_args()
    qs = [q for q in json.loads(config.QUESTIONS_JSON.read_text(encoding="utf-8"))["questions"]
          if q["split"] == args.split]
    cache: dict = {}
    out_dir = config.RESULTS / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, all_traces = [], []
    for m in args.methods:
        for uf in ([True, False] if args.no_filter else [True]):
            s, t = evaluate(m, qs, cache, uf)
            summaries.append(s)
            all_traces += t
            print(f"{args.split} {m:7s} filter={uf!s:5s} Hit@1 {s['hit1']}/{s['N']}  Hit@3 {s['hit3']}/{s['N']}  "
                  f"MRR@10 {s['mrr10']}  miss@3 {s['miss_at_3']}  {s['latency_ms_median']} ms")
    (out_dir / f"{args.split}_summary.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    (out_dir / f"{args.split}_traces.json").write_text(json.dumps(all_traces, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
