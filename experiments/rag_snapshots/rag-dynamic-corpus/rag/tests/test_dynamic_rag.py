"""Existing RAG/selectors/reranker/evaluator, with no GPU or model API calls."""
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.dynamic_corpus import DynamicCorpus, Mutation
from extensions.dynamic_corpus.pipeline import DynamicRAGPipeline, make_pipeline
from extensions.dynamic_corpus.trial import run_trial
from extensions.dynamic_corpus.workload import Catalog, Queries
from scripts.evaluate_answers import validate_candidates
from src.context_builder import SimpleContextBuilder, ExperimentalContextBuilder
from src.rag import RAGPipeline
from src.reranker import CrossEncoderReranker
from src.retriever import ExactDenseRetriever


class Generator:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return {"answer": "Mock answer, not an evaluated model result", "latency_ms": 0.0}


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.records = {"a": "Alpha cats are friendly.", "b": "Beta dogs are helpful."}
        self.matrix = np.eye(2, dtype=np.float32)
        self.index = DynamicCorpus(self.records, self.matrix)
        self.generator = Generator()
        self.common = dict(asr=SimpleNamespace(transcribe=lambda _: "Which cats are friendly?"),
                           encoder=SimpleNamespace(encode=lambda _: np.array([1, 0], np.float32)),
                           context_builder=SimpleContextBuilder(), generator=self.generator)

    def test_static_factory_preserves_original_pipeline_and_dynamic_matches_same_state(self):
        static = make_pipeline(retriever=ExactDenseRetriever.from_arrays(self.matrix, list(self.records)),
                               data=SimpleNamespace(record_text=self.records.__getitem__), **self.common)
        self.assertIs(type(static), RAGPipeline)
        dynamic = DynamicRAGPipeline(corpus=self.index, **self.common)
        a, b = static.answer_audio("fixture", k=2), dynamic.answer_audio("fixture", k=2)
        for key in ("prompt", "contexts", "selected_evidence", "answer"):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a["retrieval"]["record_ids"], b["retrieval"]["record_ids"])
        self.assertEqual(b["corpus_version"], 0)

    def test_insert_update_delete_visible_in_later_evidence(self):
        pipeline = DynamicRAGPipeline(corpus=self.index, **self.common)
        self.index.delete_document("a")
        first = pipeline.answer_audio("fixture", k=2)
        self.assertNotIn("Alpha", first["prompt"])
        self.assertNotIn("a", first["retrieval"]["record_ids"])
        self.index.insert_document("new", "Cats inserted evidence.", [1, 0])
        inserted = pipeline.answer_audio("fixture", k=1)
        self.assertEqual(inserted["retrieval"]["record_ids"], ["new"])
        self.assertIn("Cats inserted evidence.", inserted["prompt"])
        self.index.update_document("new", "Cats UPDATED evidence.", [1, 0])
        updated = pipeline.answer_audio("fixture", k=1)
        self.assertIn("Cats UPDATED evidence.", updated["prompt"])
        self.assertNotIn("inserted evidence", updated["prompt"])
        validate_candidates([{**updated, "query_id": "q", "split": "practice"}], {"q": {}}, "practice")

    def test_actual_tfidf_and_mmr_preserve_source_ids_and_updated_text(self):
        self.index.update_document("a", "Cats NEW evidence is useful.", [1, 0])
        for strategy in ("tfidf", "mmr"):
            with self.subTest(strategy=strategy):
                common = {**self.common, "context_builder": ExperimentalContextBuilder(strategy=strategy)}
                result = DynamicRAGPipeline(corpus=self.index, **common).answer_audio("fixture", k=2)
                self.assertIn("NEW evidence", result["prompt"])
                self.assertTrue(result["selected_evidence"])
                for evidence in result["selected_evidence"]:
                    self.assertIn(evidence["record_id"], {"a", "b"})
                    self.assertIn(evidence["text"], self.index.record_text(evidence["record_id"]))

    def test_existing_reranker_gets_same_snapshot_as_context_even_during_mutation(self):
        seen = []
        class Model:
            def predict(model, pairs, **kwargs):
                seen.extend(pairs)
                self.index.apply_batch([Mutation("update", "a", "Changed AFTER retrieval", [0, 1]),
                                        Mutation("delete", "b")])
                return [0.1, 0.9]
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker.model, reranker.batch_size = Model(), 8
        result = DynamicRAGPipeline(corpus=self.index, reranker=reranker, **self.common).answer_audio("fixture", k=2, final_k=1)
        self.assertEqual(result["final_retrieval"]["record_ids"], ["b"])
        self.assertEqual(result["contexts"], [self.records["b"]])
        self.assertEqual(result["corpus_version"], 0)
        self.assertEqual([pair[1] for pair in seen], list(self.records.values()))
        self.assertNotIn("b", self.index.search([1, 0], 2)["record_ids"])

    def test_parallel_pipeline_requests_have_separate_snapshot_bindings(self):
        entered, resume = threading.Event(), threading.Event()
        simple = SimpleContextBuilder()
        class Builder:
            def build(builder, text, ids, data):
                if threading.current_thread().name == "older-request":
                    entered.set()
                    if not resume.wait(5):
                        raise RuntimeError("test timed out")
                return simple.build(text, ids, data)
        pipeline = DynamicRAGPipeline(corpus=self.index, **{**self.common, "context_builder": Builder()})
        older = []
        thread = threading.Thread(name="older-request", target=lambda: older.append(pipeline.answer_audio("x", k=1)))
        thread.start()
        try:
            self.assertTrue(entered.wait(5))
            self.index.update_document("a", "Latest cats evidence", [1, 0])
            latest = pipeline.answer_audio("y", k=1)
        finally:
            resume.set()
            thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(older[0]["contexts"], [self.records["a"]])
        self.assertEqual(latest["contexts"], ["Latest cats evidence"])
        self.assertLess(older[0]["corpus_version"], latest["corpus_version"])

    def test_budget_interface_receives_dense_scores_and_forwards_metrics(self):
        class Builder:
            def build_with_retrieval(builder, text, retrieval, data):
                self.assertEqual(retrieval["scores"], [1.0, 0.0])
                return {**SimpleContextBuilder().build(text, retrieval["record_ids"], data),
                        "budget_metrics": {"fixture": True}, "discarded_evidence": []}
        result = DynamicRAGPipeline(corpus=self.index, **{**self.common, "context_builder": Builder()}).answer_audio("x", k=2)
        self.assertEqual(result["budget_metrics"], {"fixture": True})

    def test_real_budget_component_if_available(self):
        try:
            from extensions.adaptive_context_budget import BudgetedContextBuilder, FixedBudget, TokenCounter
        except ImportError:
            self.skipTest("optional teammate budget extension not merged; interface separately tested")
        builder = BudgetedContextBuilder(SimpleContextBuilder(), FixedBudget(1000), TokenCounter())
        result = DynamicRAGPipeline(corpus=self.index, **{**self.common, "context_builder": builder}).answer_audio("x", k=2)
        self.assertIn("Alpha", result["prompt"])
        self.assertIn("budget_metrics", result)

    def test_matched_trial_schema_and_generation_reuse_without_remote_calls(self):
        catalog = Catalog.synthetic(30, 4)
        # Mark this as a mock course-like fixture ONLY inside the test.
        catalog.source, catalog.embedding_space = "course_precomputed_fixture", "fixture"
        queries = Queries(["q1", "q2"], catalog.vectors[:2], "real_asr_fixture", "fixture",
                          ["Which cats?", "Which dogs?"], ["fixture.wav"]*2, "practice")
        with tempfile.TemporaryDirectory() as folder:
            rows = run_trial(catalog, queries, Path(folder)/"mock", self.generator, SimpleContextBuilder(), operations=10)
            for condition, candidates in rows.items():
                validate_candidates(candidates, {"q1": {}, "q2": {}}, "practice")
            self.assertTrue(all(r["generation_cache_hit"] for r in rows["freshly_rebuilt_after"]))
            self.assertLess(len(self.generator.prompts), 8)


if __name__ == "__main__":
    unittest.main()
