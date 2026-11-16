"""Code tests for src/ (mock model + in-memory retriever). These test control flow only;
benchmark numbers come from the real model and the Day 6 vector store."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import ABSTAIN_TEXT, Agent, AgentConfig  # noqa: E402
from src.evidence import is_hit, norm  # noqa: E402

CHUNKS = [
    {"id": "c1", "text": "A U.S. patent is only recognized domestically, and cannot be enforced in another country.",
     "source_id": "doc", "source": "doc.pdf", "page": 19, "metadata": {"section": "1.1", "ip_type": "patent"},
     "score": 0.9, "rank": 1, "method": "fake"},
    {"id": "c2", "text": "Copyright lasts for the life of the author plus 70 years.",
     "source_id": "doc", "source": "doc.pdf", "page": 116, "metadata": {"section": "3.2", "ip_type": "copyright"},
     "score": 0.5, "rank": 2, "method": "fake"},
]


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, *, filters=None, top_k=3):
        self.calls.append((query, filters, top_k))
        return [c for c in CHUNKS if not filters or c["metadata"].get("ip_type") == filters.get("ip_type")][:top_k]


def tool_resp(name, args):
    call = SimpleNamespace(id=f"x_{name}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    msg = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.seen_tools = []

    def create(self, **kw):
        self.seen_tools.append([t["function"]["name"] for t in kw["tools"]])
        return self.responses.pop(0)


def make(responses, **cfg):
    r = FakeRetriever()
    client = FakeClient(responses)
    return Agent(AgentConfig("t", **cfg), r, client=client, model="gpt-4o-mini"), r, client


class AgentTest(unittest.TestCase):
    def test_original_question_searched_first_and_answer_accepted(self):
        a, r, _ = make([tool_resp("answer", {"answer": "No [c1].", "citations": ["c1"]})])
        t = a.run("Can a U.S. patent be enforced abroad?")
        self.assertEqual(r.calls[0][0], "Can a U.S. patent be enforced abroad?")
        self.assertEqual(t["events"][0]["query_kind"], "original")
        self.assertEqual(t["status"], "answered")
        self.assertEqual(t["citations"], ["c1"])
        self.assertAlmostEqual(t["cost_usd"], 100 / 1e6 * 0.15 + 10 / 1e6 * 0.60)

    def test_citation_not_in_context_gets_one_correction_then_abstains(self):
        bad = tool_resp("answer", {"answer": "Yes [zz].", "citations": ["zz"]})
        a, _, _ = make([bad, bad])
        t = a.run("q")
        self.assertEqual(t["status"], "abstain")
        self.assertEqual(t["answer"], ABSTAIN_TEXT)
        self.assertEqual(sum(e["event"] == "invalid_or_missing_citations" for e in t["events"]), 2)

    def test_valid_id_missing_inline_is_appended_after_correction(self):
        no_inline = tool_resp("answer", {"answer": "No.", "citations": ["c1"]})
        a, _, _ = make([no_inline, no_inline])
        t = a.run("q")
        self.assertEqual(t["status"], "answered")
        self.assertIn("[c1]", t["answer"])
        self.assertTrue(any(e["event"] == "citations_appended_inline" for e in t["events"]))

    def test_retry_before_abstain_forces_one_rephrased_search(self):
        a, r, _ = make([tool_resp("abstain", {"reason": "x"}), tool_resp("search", {"query": "other words"}),
                        tool_resp("abstain", {"reason": "still none"})], max_searches=3, retry_before_abstain=True)
        t = a.run("q")
        self.assertEqual(len(r.calls), 2)
        self.assertEqual(t["status"], "abstain")
        self.assertTrue(any(e["event"] == "abstain_retry_nudge" for e in t["events"]))

    def test_search_tool_hidden_when_limit_reached(self):
        a, _, client = make([tool_resp("abstain", {"reason": "none"})], max_searches=1)
        t = a.run("q")
        self.assertNotIn("search", client.seen_tools[0])
        self.assertEqual(t["status"], "abstain")

    def test_multi_search_dedups_chunks_and_respects_filter(self):
        a, r, _ = make([tool_resp("search", {"query": "patent abroad"}),
                        tool_resp("answer", {"answer": "No [c1].", "citations": ["c1"]})], max_searches=2)
        t = a.run("q", filters={"ip_type": "patent"})
        self.assertEqual([c[1] for c in r.calls], [{"ip_type": "patent"}] * 2)
        self.assertEqual(t["events"][1]["query_kind"], "model_rewrite")
        self.assertEqual(t["events"][1]["added_to_context"], [])  # c1 already shown
        self.assertEqual([c["id"] for c in t["final_context"]], ["c1"])

    def test_budget_truncates_context(self):
        a, _, _ = make([tool_resp("abstain", {"reason": "x"})], budget_chars=250)
        t = a.run("q")
        self.assertLessEqual(t["context_chars"], 250)

    def test_llm_error_is_reported_not_faked(self):
        a, _, client = make([])
        client.chat.completions.create = lambda **kw: (_ for _ in ()).throw(RuntimeError("down"))
        t = a.run("q")
        self.assertEqual(t["status"], "error")
        self.assertIn("down", t["error"])


class EvidenceTest(unittest.TestCase):
    def test_hit_needs_page_and_quote(self):
        q = {"gold": [{"source_id": "doc", "page": 19, "evidence": "cannot be enforced in another country"}]}
        self.assertTrue(is_hit(q, CHUNKS[0]))
        self.assertFalse(is_hit(q, {**CHUNKS[0], "page": 20}))
        self.assertFalse(is_hit(q, CHUNKS[1]))

    def test_norm_quotes_and_space(self):
        self.assertEqual(norm("It’s  A\n“mark”"), "it's a \"mark\"")


class BenchmarkFileTest(unittest.TestCase):
    def test_split_sizes_and_gold_resolves(self):
        path = ROOT / "data" / "benchmark" / "questions.json"
        chunks_path = ROOT / "data" / "corpus" / "chunks.json"
        if not path.exists() or not chunks_path.exists():
            self.skipTest("run src.export_day6 and src.build_benchmark first")
        qs = json.loads(path.read_text(encoding="utf-8"))["questions"]
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        count = lambda s, a: sum(q["split"] == s and q["answerable"] == a for q in qs)
        self.assertGreaterEqual(count("dev", True), 15)
        self.assertGreaterEqual(count("dev", False), 2)
        self.assertGreaterEqual(count("test", True), 5)
        self.assertGreaterEqual(count("test", False), 2)
        for q in qs:
            if q["answerable"]:
                self.assertTrue(any(is_hit(q, c) for c in chunks), q["id"])


if __name__ == "__main__":
    unittest.main()
