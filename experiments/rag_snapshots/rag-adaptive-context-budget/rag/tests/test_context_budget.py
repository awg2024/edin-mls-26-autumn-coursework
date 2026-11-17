"""Budget correctness over ordered units, independent of retrieval/LLM services."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extensions.adaptive_context_budget.core import (
    AdaptiveBudget, EvidenceUnit, FixedBudget, TokenCounter, apply_budget, pack_evidence,
)


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.counter = TokenCounter("exact_character_fixture", len)
        self.units = [EvidenceUnit("doc-a", "First sentence."), EvidenceUnit("doc-b", "Second sentence.")]

    def test_entire_formatted_string_enforces_boundary(self):
        limit = len(self.units[0].render())
        packed = pack_evidence(self.units, limit, self.counter)
        self.assertEqual(packed["units"], self.units[:1])
        self.assertEqual(packed["tokens"], limit)

    def test_formatting_overhead_can_prevent_text_that_alone_fits(self):
        self.assertEqual(pack_evidence(self.units, len(self.units[0].text), self.counter)["units"], [])

    def test_separator_is_counted(self):
        limit = sum(len(u.render()) for u in self.units)
        self.assertEqual(len(pack_evidence(self.units, limit, self.counter)["units"]), 1)
        self.assertEqual(len(pack_evidence(self.units, limit + 2, self.counter)["units"]), 2)

    def test_empty_and_tiny_budgets(self):
        self.assertEqual(pack_evidence([], 0)["evidence"], "")
        for budget in (0, 1, 2, 3):
            with self.subTest(budget=budget):
                result = pack_evidence(self.units, budget, self.counter)
                self.assertLessEqual(result["tokens"], budget)
                self.assertEqual(result["units"], [])

    def test_oversized_is_skipped_and_later_unit_can_fit(self):
        large = EvidenceUnit("large", "x" * 1000)
        result = pack_evidence([large, self.units[0]], len(self.units[0].render()), self.counter)
        self.assertEqual(result["units"], self.units[:1])
        self.assertEqual(result["discarded"][0]["reason"], "over_budget")

    def test_exact_duplicate_text_keeps_first_source(self):
        duplicate = EvidenceUnit("other-source", " first   SENTENCE. ")
        result = pack_evidence([self.units[0], duplicate, self.units[1]], 1000, self.counter)
        self.assertEqual(result["units"], self.units)
        self.assertEqual(result["discarded"][0]["reason"], "duplicate")

    def test_accepted_order_is_never_reranked(self):
        result = pack_evidence(self.units[::-1], 1000, self.counter)
        self.assertEqual([u.source_id for u in result["units"]], ["doc-b", "doc-a"])

    def test_packing_is_deterministic_and_does_not_mutate_inputs(self):
        before = list(self.units)
        result = pack_evidence(self.units, 1000, self.counter)
        self.assertEqual(result, pack_evidence(self.units, 1000, self.counter))
        self.assertEqual(before, self.units)

    def test_sentence_prefix_ends_at_complete_sentence(self):
        unit = EvidenceUnit("doc", "Short. " + "An oversized second sentence " * 50 + ".")
        limit = len(EvidenceUnit("doc", "Short.").render())
        result = pack_evidence([unit], limit, self.counter, "sentence_prefix")
        self.assertEqual(result["units"][0].text, "Short.")
        self.assertEqual(result["truncated_units"], 1)
        self.assertEqual(result["units"][0].source_id, "doc")

    def test_no_arbitrary_word_truncation(self):
        result = pack_evidence([EvidenceUnit("doc", "no punctuation " * 100)], 30,
                               self.counter, "sentence_prefix")
        self.assertEqual(result["units"], [])

    def test_truncated_prefix_does_not_duplicate_earlier_evidence(self):
        units = [EvidenceUnit("a", "Short."), EvidenceUnit("b", "Short. " + "Long " * 100 + ".")]
        result = pack_evidence(units, 50, self.counter, "sentence_prefix")
        self.assertEqual(len(result["units"]), 1)

    def test_all_selected_dimensions_and_overhead_count_against_budget(self):
        for policy in ("skip", "sentence_prefix"):
            for budget in range(0, 100):
                result = pack_evidence(self.units, budget, self.counter, policy)
                self.assertLessEqual(len(result["evidence"]), budget)

    def test_unicode_default_counts_utf8_bytes(self):
        text = "é 東京 🙂"
        self.assertEqual(TokenCounter().count(text), (len(text.encode("utf-8")) + 3) // 4)

    def test_source_and_custom_label_preserved(self):
        unit = EvidenceUnit("actual/id:7", "Evidence.", unit_id="4", label="2", metadata={"relevance_score": 0.7})
        result = pack_evidence([unit], 100)
        self.assertIn("[Document 2]", result["evidence"])
        self.assertEqual(result["units"][0].as_dict()["record_id"], "actual/id:7")
        self.assertEqual(result["units"][0].as_dict()["relevance_score"], 0.7)

    def test_invalid_inputs(self):
        for budget in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                pack_evidence(self.units, budget)
        with self.assertRaises(ValueError):
            pack_evidence(self.units, 5, packing_policy="rank_again")
        with self.assertRaises(ValueError):
            EvidenceUnit("", "text")
        with self.assertRaises(ValueError):
            EvidenceUnit("bad\nlabel", "text")
        with self.assertRaises(ValueError):
            TokenCounter(function=lambda text: True).count("a")

    def test_zero_denominators_are_explicitly_missing(self):
        metrics = apply_budget([], None, FixedBudget(0))["metrics"]
        self.assertIsNone(metrics["token_reduction_pct"])
        self.assertIsNone(metrics["budget_utilisation"])


class AdaptiveTests(unittest.TestCase):
    units = [EvidenceUnit("a", "Evidence.")]

    def test_clear_gap_gets_minimum(self):
        self.assertEqual(AdaptiveBudget(100, 900).decide(self.units, [1, 0.2, 0.1]).budget, 100)

    def test_ties_and_near_ties_get_maximum(self):
        for scores in ([0.5] * 5, [0.9, 0.89, 0.2], [-1, -1, -1]):
            self.assertEqual(AdaptiveBudget(100, 900).decide(self.units, scores).budget, 900)

    def test_middle_confidence_maps_inside_bounds(self):
        decision = AdaptiveBudget(100, 900).decide(self.units, [1, 0.8, 0.2])
        self.assertAlmostEqual(decision.confidence, 0.25)
        self.assertEqual(decision.budget, 660)

    def test_positive_affine_score_changes_do_not_change_budget(self):
        original = [-0.1, -0.2, -0.7, -0.9]
        policy = AdaptiveBudget()
        self.assertEqual(policy.decide(self.units, original).budget,
                         policy.decide(self.units, [7 * x + 20 for x in original]).budget)

    def test_missing_partial_nonfinite_scores_fallback_conservatively(self):
        for scores in (None, [], [1], [1, 0], [1, None, 0], [1, float("nan"), 0], [1, float("inf"), 0]):
            with self.subTest(scores=scores):
                self.assertEqual(AdaptiveBudget(100, 900).decide(self.units, scores).budget, 900)

    def test_empty_candidates_and_equal_bounds(self):
        self.assertEqual(AdaptiveBudget(100, 900).decide([], [1, 0, -1]).budget, 100)
        self.assertEqual(AdaptiveBudget(123, 123).decide(self.units, [1, 0, -1]).budget, 123)

    def test_finite_extreme_scores_do_not_overflow(self):
        decision = AdaptiveBudget().decide(self.units, [1e308, 0, -1e308])
        self.assertTrue(256 <= decision.budget <= 2048)

    def test_invalid_policy_configuration(self):
        for kwargs in ({"min_budget": -1}, {"min_budget": 10, "max_budget": 1},
                       {"low_confidence": 0.7, "high_confidence": 0.5}, {"gamma": 0}):
            with self.assertRaises(ValueError):
                AdaptiveBudget(**kwargs)


if __name__ == "__main__":
    unittest.main()
