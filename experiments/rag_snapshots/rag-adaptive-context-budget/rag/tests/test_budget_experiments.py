"""Controlled comparisons and evaluator-schema compatibility; no real API calls."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.adaptive_context_budget.analysis import load_run, paired_interval, plot_results, summarise
from extensions.adaptive_context_budget.cli import main
from extensions.adaptive_context_budget.core import AdaptiveBudget, FixedBudget, TokenCounter
from extensions.adaptive_context_budget.experiment import (
    Condition, evaluate_condition, make_builders, make_conditions, match_fixed_budget,
    prepare_query, render_condition, run_comparison, validate_snapshot,
)
from src.evaluation.answer_judge import write_jsonl_atomic
from scripts.evaluate_answers import validate_candidates
from test_budget_integration import FixtureData, MockEncoder, MockGenerator, MockRetriever


def snapshot(selectors=("simple", "tfidf", "mmr")):
    builders = make_builders(selectors)
    return [prepare_query({"query_id": f"q{i}", "split": "practice", "audio_file": f"q{i}.wav"},
                          f"How does cache memory affect latency? Query {i}.",
                          FixtureData(), MockEncoder(), MockRetriever(), builders) for i in range(2)]


class ExperimentTests(unittest.TestCase):
    def test_frozen_candidates_and_selector_outputs_identical_for_all_conditions(self):
        queries = snapshot()
        saved = deepcopy(queries)
        counter = TokenCounter()
        specs = [Condition("mmr_none", "mmr"), Condition("mmr_fixed_20_skip", "mmr", FixedBudget(20))]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            results = run_comparison(queries, specs, output, counter, MockGenerator())
            for left, right in zip(results[specs[0].name], results[specs[1].name]):
                self.assertEqual(left["transcript"], right["transcript"])
                self.assertEqual(left["candidate_ranking_hash"], right["candidate_ranking_hash"])
                self.assertEqual(left["selector_evidence_hash"], right["selector_evidence_hash"])
                self.assertNotEqual(left["prompt"], right["prompt"])
            self.assertEqual(queries, saved)

    def test_each_existing_selector_runs_once_during_preparation(self):
        calls = []
        class RecordingBuilder:
            def build(self, query, ids, data):
                calls.append(tuple(ids))
                return make_builders(["simple"])["simple"].build(query, ids, data)
        query = prepare_query({"query_id": "q", "split": "practice"}, "Question?", FixtureData(),
                              MockEncoder(), MockRetriever(), {"simple": RecordingBuilder()})
        for budget in (20, 40, 60, 80):
            render_condition(query, Condition(str(budget), "simple", FixedBudget(budget)), TokenCounter())
        self.assertEqual(len(calls), 1)

    def test_mock_generated_outputs_pass_supplied_candidate_validator(self):
        queries = snapshot(["simple"])
        spec = Condition("simple_fixed_40_skip", "simple", FixedBudget(40))
        rows = [evaluate_condition(q, spec, TokenCounter(), MockGenerator()) for q in queries]
        self.assertEqual(validate_candidates(rows, {q["query_id"]: {} for q in queries}, "practice"), {"q0", "q1"})
        self.assertEqual(rows[0]["api_input_tokens"], 123)
        self.assertGreater(rows[0]["estimated_prompt_text_tokens"], rows[0]["evidence_tokens_after"])
        self.assertIsNone(rows[0]["timing"]["end_to_end_ms"])

    def test_offline_results_never_fabricate_answers_or_api_usage(self):
        query = snapshot(["simple"])[0]
        row = evaluate_condition(query, Condition("simple_none", "simple"), TokenCounter())
        self.assertNotIn("answer", row)
        self.assertIsNone(row["api_input_tokens"])
        self.assertIsNone(row["timing"]["generation_ms"])
        self.assertEqual(row["generation_status"], "not_requested")

    def test_qrels_only_change_evaluation_not_budget_or_evidence(self):
        query = snapshot(["mmr"])[0]
        spec = Condition("adaptive", "mmr", AdaptiveBudget(10, 30))
        a = evaluate_condition(query, spec, TokenCounter(), qrels={"q0": {"doc-a": 1}})
        b = evaluate_condition(query, spec, TokenCounter(), qrels={"q0": {"doc-c": 1}})
        self.assertEqual(a["configured_budget"], b["configured_budget"])
        self.assertEqual(a["prompt"], b["prompt"])

    def test_query_confidence_strata_are_shared_by_budgeted_and_control_rows(self):
        query = snapshot(["simple"])[0]
        control = evaluate_condition(query, Condition("simple_none", "simple"), TokenCounter())
        budgeted = evaluate_condition(query, Condition("fixed", "simple", FixedBudget(20)), TokenCounter())
        self.assertIsNotNone(control["query_retrieval_confidence"])
        self.assertEqual(control["query_retrieval_confidence"], budgeted["query_retrieval_confidence"])

    def test_missing_relevant_control_documents_give_missing_retention(self):
        query = snapshot(["simple"])[0]
        row = evaluate_condition(query, Condition("none", "simple"), TokenCounter(), qrels={"q0": {"absent": 1}})
        self.assertIsNone(row["relevant_document_retention"])

    def test_used_token_matching_is_deterministic_and_does_not_use_quality(self):
        queries = snapshot(["simple"])
        spec = Condition("adaptive", "simple", AdaptiveBudget(40, 100))
        first = match_fixed_budget(queries, spec, TokenCounter(), 100)
        self.assertEqual(first, match_fixed_budget(queries, spec, TokenCounter(), 100))
        self.assertAlmostEqual(first[1], first[2])

    def test_condition_factory_contains_same_selector_controls_and_matching(self):
        specs = make_conditions(snapshot(["mmr"]), budgets=[32], packing_policies=["skip"], match_adaptive=True)
        self.assertEqual({s.name for s in specs}, {"mmr_none", "mmr_fixed_32_skip", "mmr_adaptive_skip", "mmr_matched_skip"})
        with self.assertRaises(ValueError):
            make_conditions(snapshot(["simple"]), budgets=[32, 32])

    def test_snapshot_tampering_and_duplicates_rejected(self):
        queries = snapshot(["simple"])
        queries[0]["final_retrieval"]["scores"][0] = 0.99
        with self.assertRaises(ValueError):
            validate_snapshot(queries)
        query = snapshot(["simple"])[0]
        with self.assertRaises(ValueError):
            validate_snapshot([query, query])

    def test_generation_failure_recorded_without_fake_candidate_answer(self):
        class BrokenGenerator:
            def generate(self, prompt):
                raise RuntimeError("fixture failure")
        row = evaluate_condition(snapshot(["simple"])[0], Condition("none", "simple"), TokenCounter(), BrokenGenerator())
        self.assertEqual(row["generation_status"], "failed")
        self.assertNotIn("answer", row)

    def test_output_overwrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileExistsError):
                run_comparison(snapshot(["simple"]), [Condition("none", "simple")], folder)

    def test_large_api_sweep_requires_explicit_opt_in_before_client_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "snapshot.jsonl"
            write_jsonl_atomic(path, snapshot(["simple"]))
            argv = ["run", "--snapshot", str(path), "--output", str(Path(folder)/"out"),
                    "--generate", "--conditions", "simple_none", "--limit", "6"]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as context:
                main(argv)
            self.assertEqual(context.exception.code, 2)

    def test_prepare_then_offline_cli_runs_with_mock_encoder_and_cached_transcripts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/"data"
            (data/"embeddings").mkdir(parents=True)
            write_jsonl_atomic(data/"corpus.jsonl", [
                {"record_id": rid, "text": text} for rid, text in FixtureData.records.items()])
            write_jsonl_atomic(data/"student_audio_manifest.jsonl", [
                {"query_id": f"q{i}", "split": "practice", "audio_file": f"q{i}.wav"} for i in range(2)])
            write_jsonl_atomic(root/"transcripts.jsonl", [
                {"query_id": f"q{i}", "transcript": "How does cache memory affect latency?"} for i in range(2)])
            # No real encoder/embedding payload is loaded in this protocol test.
            (data/"embeddings/corpus_enc_a.npy").touch()
            (data/"embeddings/corpus_enc_a_ids.json").write_text("[]")
            snap = root/"frozen.jsonl"
            with patch("src.encoder.LocalEncoder", side_effect=lambda *a, **k: MockEncoder()), \
                 patch("src.retriever.ExactDenseRetriever", side_effect=lambda *a, **k: MockRetriever()), \
                 redirect_stdout(io.StringIO()):
                main(["prepare", "--data-dir", str(data), "--transcript-cache", str(root/"transcripts.jsonl"),
                      "--output", str(snap), "--limit", "2"])
                main(["run", "--snapshot", str(snap), "--output", str(root/"run"),
                      "--budgets", "32", "--packing-policies", "skip", "--match-adaptive",
                      "--conditions", "simple_none", "simple_fixed_32_skip", "mmr_none", "mmr_adaptive_skip", "mmr_matched_skip"])
            self.assertTrue((root/"frozen.lengths.csv").is_file())
            self.assertTrue((root/"run/summary.csv").is_file())
            self.assertEqual(list((root/"run").glob("*/candidates.jsonl")), [])
            manifest = json.loads((root/"run/run_manifest.json").read_text())
            self.assertFalse(manifest["generation_requested"])
            self.assertEqual(manifest["provenance"]["k"], 5)

    def test_paired_scores_coverage_and_confidence_intervals(self):
        queries = snapshot(["simple"])
        specs = [Condition("simple_none", "simple"), Condition("simple_fixed_30_skip", "simple", FixedBudget(30))]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            run_comparison(queries, specs, output, generator=MockGenerator())
            for name, value in (("simple_none", 3), ("simple_fixed_30_skip", 2)):
                scores = [{"query_id": q["query_id"], "correctness": value, "grounding": value,
                           "completeness": value, "normalised_answer_score": value/4} for q in queries]
                write_jsonl_atomic(output/name/"judge/answer_judge_per_query.jsonl", scores)
            summary = summarise(output)
            self.assertTrue(all(s["quality_coverage"] == 1 for s in summary))
            import csv
            with (output/"paired_summary.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            quality = next(r for r in pairs if r["metric"] == "normalised_answer_score")
            self.assertAlmostEqual(float(quality["mean_delta"]), -0.25)
            self.assertAlmostEqual(float(quality["bootstrap_95_low"]), -0.25)
            self.assertEqual(int(quality["paired_queries"]), 2)

    def test_partial_quality_coverage_is_not_hidden(self):
        queries = snapshot(["simple"])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            run_comparison(queries, [Condition("simple_none", "simple")], output)
            write_jsonl_atomic(output/"simple_none/judge/answer_judge_per_query.jsonl", [
                {"query_id": "q0", "correctness": 0, "grounding": 0, "completeness": 0, "normalised_answer_score": 0}])
            summary = summarise(output)[0]
            self.assertEqual(summary["quality_coverage"], 0.5)
            self.assertEqual(summary["normalised_answer_score_n"], 1)
            self.assertIsNone(summary["normalised_answer_score_sd"])
        self.assertEqual(paired_interval([0.5]), (None, None))

    def test_pairing_rejects_different_candidate_rankings(self):
        queries = snapshot(["simple"])
        specs = [Condition("simple_none", "simple"), Condition("simple_fixed_20_skip", "simple", FixedBudget(20))]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            run_comparison(queries, specs, output)
            path = output/"simple_fixed_20_skip/metrics.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["candidate_ranking_hash"] = "changed"
            write_jsonl_atomic(path, rows)
            with self.assertRaises(ValueError):
                summarise(output)

    def test_original_baseline_comparisons_are_labelled_separately_from_budget_effects(self):
        import csv
        queries = snapshot(["simple", "mmr"])
        specs = [Condition("simple_none", "simple"), Condition("mmr_none", "mmr"),
                 Condition("mmr_fixed_20_skip", "mmr", FixedBudget(20))]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            run_comparison(queries, specs, output)
            summarise(output)
            with (output/"paired_summary.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            kinds = {row["comparison_type"] for row in pairs}
            self.assertEqual(kinds, {"budget_only", "selector_and_budget_practical"})

    def test_plots_skip_missing_quality_instead_of_fabricating_points(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            run_comparison(snapshot(["simple"]), [Condition("simple_none", "simple")], output)
            made = plot_results(Path(folder)/"plots", run_dir=output)
            self.assertEqual(made, [])
            manifest = json.loads((Path(folder)/"plots/plot_manifest.json").read_text())
            self.assertIn("quality_vs_budget", manifest["skipped"])

    def test_all_six_plot_types_accept_evaluator_schema_with_temporary_mock_data(self):
        # Figures in this test are temporary fixtures, never published as measurements.
        queries = snapshot(["simple"])
        specs = [Condition("simple_none", "simple"),
                 Condition("simple_fixed_30_skip", "simple", FixedBudget(30)),
                 Condition("simple_adaptive_skip", "simple", AdaptiveBudget(30, 100)),
                 Condition("simple_matched_skip", "simple", FixedBudget(50), "skip", 40)]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"run"
            run_comparison(queries, specs, output, generator=MockGenerator())
            for spec in specs:
                write_jsonl_atomic(output/spec.name/"judge/answer_judge_per_query.jsonl", [
                    {"query_id": q["query_id"], "correctness": 2, "grounding": 2,
                     "completeness": 2, "normalised_answer_score": 0.5} for q in queries])
            overhead = Path(folder)/"fixture_overhead.csv"
            overhead.write_text("configuration,candidate_units,mean_ms,sd_ms\nfixture,1,0.1,0.01\nfixture,2,0.2,0.01\n")
            made = plot_results(Path(folder)/"plots", output, overhead)
            self.assertEqual(len(made), 6)
            for name in made:
                for suffix in ("png", "pdf"):
                    self.assertTrue((Path(folder)/"plots"/f"{name}.{suffix}").is_file())


if __name__ == "__main__":
    unittest.main()
