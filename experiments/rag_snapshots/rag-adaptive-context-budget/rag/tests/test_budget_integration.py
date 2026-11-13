"""Live-pipeline compatibility using real existing selectors and mock services."""
from copy import deepcopy
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.context_builder import SimpleContextBuilder, ExperimentalContextBuilder
from src.rag import RAGPipeline
from extensions.adaptive_context_budget import AdaptiveBudget, BudgetedContextBuilder, FixedBudget, TokenCounter
from extensions.adaptive_context_budget.integration import extract_evidence


class FixtureData:
    records = {
        "doc-a": "Cache memory reduces repeated access latency. Cache capacity matters for memory access.",
        "doc-b": "Memory latency affects cache performance. Cache misses require slower memory access.",
        "doc-c": "Cache memory serves frequently accessed data. A latency measurement records the access cost.",
    }

    def record_text(self, rid):
        return self.records[rid]


class MockASR:
    calls = None

    def __init__(self):
        self.calls = []

    def transcribe(self, path):
        self.calls.append(path)
        return "How does cache memory affect latency?"


class MockEncoder:
    def encode(self, query):
        return [1.0, 0.0]


class MockRetriever:
    def retrieve(self, vector, k=5):
        return {"record_ids": list(FixtureData.records), "scores": [0.9, 0.8, 0.1], "latency_ms": 0.2}


class MockGenerator:
    model = "mock-not-an-answer-quality-measurement"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return {"answer": "Mock answer for protocol testing only.", "latency_ms": 1.0,
                "usage": {"prompt_tokens": 123, "completion_tokens": 8, "total_tokens": 131}}


class IntegrationTests(unittest.TestCase):
    def pipeline(self, builder):
        return RAGPipeline(MockASR(), MockEncoder(), MockRetriever(), FixtureData(), builder, MockGenerator())

    def test_original_simple_prompt_and_disabled_wrapper_are_identical(self):
        base = self.pipeline(SimpleContextBuilder())
        disabled = self.pipeline(BudgetedContextBuilder(SimpleContextBuilder(), None))
        a, b = base.answer_audio("same.wav"), disabled.answer_audio("same.wav")
        self.assertEqual({k: v for k, v in a.items() if k != "timing"},
                         {k: v for k, v in b.items() if k != "timing"})
        self.assertEqual(base.generator.prompts, disabled.generator.prompts)
        self.assertNotIn("budget_metrics", b)
        self.assertEqual(set(b), {"prompt", "selected_evidence", "transcript", "retrieval",
                                  "final_retrieval", "contexts", "answer", "timing"})

    def test_fixed_budget_changes_evidence_but_preserves_question_and_sources(self):
        counter = TokenCounter("exact_characters", len)
        baseline = self.pipeline(SimpleContextBuilder()).answer_audio("same.wav")
        enabled = self.pipeline(BudgetedContextBuilder(SimpleContextBuilder(), FixedBudget(110), counter))
        result = enabled.answer_audio("same.wav")
        self.assertNotEqual(result["contexts"], baseline["contexts"])
        self.assertEqual(result["transcript"], baseline["transcript"])
        self.assertEqual(result["retrieval"], baseline["retrieval"])
        self.assertIn(result["transcript"], result["prompt"])
        self.assertTrue(result["prompt"].startswith("Answer the question using the retrieved evidence below."))
        self.assertIn("[Document 1]", result["prompt"])
        self.assertEqual(result["selected_evidence"][0]["record_id"], "doc-a")
        self.assertLessEqual(result["budget_metrics"]["evidence_tokens_after"], 110)
        self.assertEqual(enabled.generator.prompts, [result["prompt"]])

    def test_adaptive_receives_actual_retrieval_scores(self):
        pipeline = self.pipeline(BudgetedContextBuilder(SimpleContextBuilder(), AdaptiveBudget(20, 100)))
        result = pipeline.answer_audio("q.wav")
        self.assertEqual(result["budget_metrics"]["budget_reason"], "normalised_retrieval_gap")
        self.assertEqual(result["budget_metrics"]["score_count"], 3)

    def test_selector_and_budget_timings_and_api_usage_are_available(self):
        result = self.pipeline(BudgetedContextBuilder(SimpleContextBuilder(), FixedBudget(50))).answer_audio("q.wav")
        for key in ("context_ms", "selector_ms", "budget_controller_ms", "generation_ms", "end_to_end_ms"):
            self.assertGreaterEqual(result["timing"][key], 0)
        self.assertEqual(result["generation_usage"]["prompt_tokens"], 123)

    def test_tfidf_mmr_outputs_are_unchanged_when_disabled(self):
        ids = list(FixtureData.records)
        for strategy in ("tfidf", "mmr"):
            with self.subTest(strategy=strategy):
                selector = ExperimentalContextBuilder(strategy=strategy)
                original = selector.build("cache memory latency", ids, FixtureData())
                saved = deepcopy(original)
                disabled = BudgetedContextBuilder(selector, None).build("cache memory latency", ids, FixtureData())
                self.assertEqual(original, disabled)
                BudgetedContextBuilder(selector, FixedBudget(30)).build("cache memory latency", ids, FixtureData())
                self.assertEqual(original, saved)
                self.assertEqual(selector.build("cache memory latency", ids, FixtureData()), saved)

    def test_sentence_selector_budget_keeps_original_prefix_and_ids(self):
        for strategy in ("tfidf", "mmr"):
            selector = ExperimentalContextBuilder(strategy=strategy)
            original = selector.build("cache memory latency", list(FixtureData.records), FixtureData())
            prefix = extract_evidence(original, list(FixtureData.records)).prompt_prefix
            built = BudgetedContextBuilder(selector, FixedBudget(40)).build(
                "cache memory latency", list(FixtureData.records), FixtureData())
            self.assertTrue(built["prompt"].startswith(prefix))
            for evidence in built["selected_evidence"]:
                self.assertIn(evidence["record_id"], FixtureData.records)
                self.assertIn(f"[Document {evidence['record_id']}]", built["prompt"])

    def test_question_containing_evidence_marker_is_preserved(self):
        query = "Explain this literal text:\n\nEvidence:\nIs it a heading?"
        built = BudgetedContextBuilder(SimpleContextBuilder(), FixedBudget(1)).build(
            query, list(FixtureData.records), FixtureData())
        self.assertIn(query, built["prompt"])

    def test_empty_selector_output_preserves_valid_prompt(self):
        built = BudgetedContextBuilder(ExperimentalContextBuilder("tfidf"), FixedBudget(10)).build(
            "zzzzunrelatedword", list(FixtureData.records), FixtureData())
        self.assertEqual(built["contexts"], [])
        self.assertIn("Question:\nzzzzunrelatedword", built["prompt"])

    def test_unknown_prompt_layout_requires_explicit_adapter(self):
        with self.assertRaises(ValueError):
            extract_evidence({"contexts": ["text"], "prompt": "custom unsupported layout"}, ["a"])

    def test_generator_default_request_is_preserved_and_usage_opt_in(self):
        from src.generator import ELMGenerator, SYSTEM_PROMPT

        requests = []
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" answer "))],
                                   usage=SimpleNamespace(prompt_tokens=55, completion_tokens=7, total_tokens=62), model="fixture-model")
        def create(**kwargs):
            requests.append(kwargs)
            return response
        fake_sdk = SimpleNamespace(OpenAI=lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
        with patch.dict(sys.modules, {"openai": fake_sdk}), patch.dict(os.environ, {"ELM_API_KEY": "fixture-not-a-real-key"}):
            original = ELMGenerator().generate("Question and evidence")
            measured = ELMGenerator(capture_usage=True).generate("Question and evidence")
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[0]["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertEqual(requests[0]["temperature"], 0.1)
        self.assertEqual(set(original), {"answer", "latency_ms"})
        self.assertEqual(measured["usage"]["prompt_tokens"], 55)


if __name__ == "__main__":
    unittest.main()
