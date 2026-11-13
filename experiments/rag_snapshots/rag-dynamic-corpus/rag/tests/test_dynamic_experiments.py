"""Validate replay, availability-aware quality and benchmark accounting."""
import contextlib
import csv
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.dynamic_corpus import DynamicCorpus
from extensions.dynamic_corpus.benchmark import Case, run_suite, suite_cases, validate_checkpoint
from extensions.dynamic_corpus.cli import main
from extensions.dynamic_corpus.plots import summarise
from extensions.dynamic_corpus.workload import (
    Catalog, Queries, apply_oracle, make_mutation_log, mutation_from_entry,
)


class ExperimentTests(unittest.TestCase):
    def test_deterministic_logs_valid_replay_and_source_immutability(self):
        catalog = Catalog.synthetic(100, 8)
        initial = catalog.initial_ids()
        before = catalog.fingerprint()
        log = make_mutation_log(catalog, initial, 100)
        self.assertEqual(log, make_mutation_log(catalog, initial, 100))
        self.assertEqual([sum(e["kind"] == kind for e in log) for kind in ("insert", "update", "delete")], [40, 30, 30])
        digests = []
        for batch_size in (1, 10, 100):
            index = DynamicCorpus({rid: catalog.records[rid] for rid in initial},
                                  catalog.vectors[[catalog.positions[r] for r in initial]], initial,
                                  embedding_space=catalog.embedding_space)
            oracle = {rid: (catalog.records[rid], catalog.vector(rid)) for rid in initial}
            sources = {rid: rid for rid in initial}
            for start in range(0, len(log), batch_size):
                index.apply_batch([mutation_from_entry(catalog, e) for e in log[start:start+batch_size]])
                for entry in log[start:start+batch_size]:
                    apply_oracle(oracle, sources, catalog, entry)
            checks = validate_checkpoint(index, oracle, sources, catalog, Queries.probes(catalog), 10)
            self.assertTrue(all(c["exact_topk_agreement"] and c["logical_state_equal"] for c in checks))
            digests.append(index.snapshot().digest())
        self.assertEqual(len(set(digests)), 1)
        self.assertEqual(catalog.fingerprint(), before)

    def test_pool_exhaustion_is_explicit(self):
        catalog = Catalog.synthetic(10, 3)
        with self.assertRaisesRegex(ValueError, "exhausts"):
            make_mutation_log(catalog, catalog.initial_ids(), 3, "insert")

    def test_qrels_distinguish_availability_and_invalid_semantic_replacements(self):
        catalog = Catalog({"a": "A", "b": "B"}, np.eye(2, dtype=np.float32), ["a", "b"], "space", "fixture")
        index = DynamicCorpus({"a": "A"}, np.array([[1, 0]], np.float32))
        oracle, sources = {"a": ("A", catalog.vector("a"))}, {"a": "a"}
        queries = Queries(["q"], np.array([[1, 0]], np.float32), "judged_fixture", "space")
        row = validate_checkpoint(index, oracle, sources, catalog, queries, 10, {"q": {"a": 1, "b": 1}})[0]
        self.assertEqual(row["original_qrels_recall_at10"], 0.5)
        self.assertEqual(row["active_qrels_recall_at10"], 1)
        self.assertEqual(row["active_qrels_mrr_at10"], 1)
        self.assertEqual(row["active_qrels_ndcg_at10"], 1)
        self.assertEqual(row["relevant_document_availability"], 0.5)
        row = validate_checkpoint(index, oracle, {"a": "b"}, catalog, queries, 10, {"q": {"a": 1}})[0]
        self.assertIsNone(row["active_qrels_recall_at10"])
        self.assertIn("invalidate", row["quality_scope"])

    def test_runner_accounting_raw_evidence_and_replay(self):
        catalog = Catalog.synthetic(60, 8)
        case = Case(corpus_size=60, operations=10, checkpoint_every=5)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder)/"measured"
            run_suite(catalog, Queries.probes(catalog, 3), [case], output, repeats=1)
            rows = summarise(output)
            self.assertEqual(len(rows), 3)
            checks = json.loads((output/"correctness_summary.json").read_text())
            self.assertEqual(checks["agreement_fraction"], 1)
            self.assertEqual(checks["qrels_evaluated_checks"], 0)
            with (output/case.name/"query_timings.csv").open() as stream:
                queries = [r for r in csv.DictReader(stream) if r["in_workload"] == "True"]
            self.assertEqual(len(queries), 30)
            self.assertEqual(len(set(q["query_id"] for q in queries)), 3)
            with (output/"all_summary_per_repeat.csv").open() as stream:
                for row in csv.DictReader(stream):
                    self.assertAlmostEqual(float(row["measured_workload_ms"]),
                                           float(row["total_mutation_ms"]) + float(row["query_time_sum_ms"]))
            main(["replay", "--dataset", "synthetic", "--synthetic-records", "60", "--dimension", "8",
                  "--case-dir", str(output/case.name)])
            with self.assertRaises(FileExistsError):
                run_suite(catalog, Queries.probes(catalog), [case], output, repeats=1)

    def test_all_required_factor_levels_and_single_query_ratio_count(self):
        cases = suite_cases()
        self.assertEqual(len(cases), 21)
        for family, key, values in (("batch_size", "batch_size", {1, 10, 100, 500}),
                                    ("query_ratio", "queries_per_update", {1, 10, 100}),
                                    ("threshold", "compaction_threshold", {.01, .05, .1, .2}),
                                    ("corpus_size", "corpus_size", {2000, 5000, 10000, 24000})):
            self.assertEqual({getattr(c, key) for c in cases if c.family == family}, values)

    def test_cli_no_assets_no_calls_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main(["benchmark", "--data-dir", folder, "--output", folder+"/out"])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse((Path(folder)/"out").exists())
            with self.assertRaises(SystemExit):
                main(["rag-trial", "--dataset", "synthetic", "--output", folder+"/out"])
            self.assertFalse((Path(folder)/"out").exists())

    def test_query_cache_roundtrip_and_space_mismatch(self):
        catalog = Catalog.synthetic(30, 3)
        queries = Queries.probes(catalog)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"queries.npz"
            np.savez_compressed(path, vectors=queries.vectors, query_ids=np.asarray(queries.ids))
            path.with_suffix(".meta.json").write_text(json.dumps({"source": queries.source, "embedding_space": queries.embedding_space}))
            loaded = Queries.load(path)
            np.testing.assert_array_equal(queries.vectors, loaded.vectors)
            loaded.embedding_space = "different"
            with self.assertRaisesRegex(ValueError, "space"):
                run_suite(catalog, loaded, [Case(corpus_size=30, operations=10)], Path(folder)/"out")


if __name__ == "__main__":
    unittest.main()
