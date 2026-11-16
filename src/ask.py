"""Ask the agent one question and print the trace (demo).

Run: python -m src.ask "Can a U.S. patent be enforced in another country?" [--config rerank] [--filter ip_type=patent]
"""
from __future__ import annotations

import argparse

from .agent import Agent
from .retrievers import build
from .run_agent import CONFIGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--config", default="rerank", choices=list(CONFIGS))
    ap.add_argument("--filter", help="e.g. ip_type=patent (values: patent, copyright, trademark, trade_secret)")
    args = ap.parse_args()
    flt = dict([args.filter.split("=", 1)]) if args.filter else None
    cfg = CONFIGS[args.config]
    t = Agent(cfg, build(cfg.method, {})).run(args.question, filters=flt)
    print(f"\n=== config: {cfg.name} ({cfg.method}, top_k={cfg.top_k}, max_searches={cfg.max_searches})")
    for e in t["events"]:
        if e["event"] == "search":
            print(f"[search/{e['query_kind']}] {e['query']!r}  filter={e['filter']}  ({e['latency_ms']} ms)")
            for r in e["results"]:
                print(f"    #{r['rank']} {r['id']}  page {r['page']}  score {r['score']:.3f}")
        else:
            print(f"[{e['event']}] " + str({k: v for k, v in e.items() if k != 'event'})[:200])
    print(f"\nSTATUS : {t['status']}\nANSWER : {t['answer']}\nCITES  : {t['citations']}")
    for c in t["final_context"]:
        if c["id"] in t["citations"]:
            print(f"\n--- [{c['id']}] page {c['page']}\n{c['text'][:600]}")
    print(f"\ntokens in/out {t['usage']['prompt_tokens']}/{t['usage']['completion_tokens']}  "
          f"cost ${t['cost_usd']}  time {t['latency_s']} s")


if __name__ == "__main__":
    main()
