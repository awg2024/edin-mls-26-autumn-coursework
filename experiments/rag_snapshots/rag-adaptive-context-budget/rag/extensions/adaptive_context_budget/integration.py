"""Adapters for the supplied Simple/ExperimentalContextBuilder result schema."""
from __future__ import annotations

from dataclasses import dataclass
import time

from .core import EvidenceUnit, TokenCounter, apply_budget, render_evidence


@dataclass(frozen=True)
class EvidenceBatch:
    units: tuple[EvidenceUnit, ...]
    prompt_prefix: str
    original_prompt: str


def extract_evidence(built: dict, record_ids) -> EvidenceBatch:
    """Keep the exact original instruction/question prefix and original labels.

    Sentence selector outputs already contain provenance. The original baseline
    labels whole documents by ordinal, so preserve those labels while retaining
    actual record IDs in structured output. Unknown formats fail explicitly;
    external selectors can supply evidence_units and prompt_prefix themselves.
    """
    if "evidence_units" in built and "prompt_prefix" in built:
        units = tuple(built["evidence_units"])
        if not all(isinstance(u, EvidenceUnit) for u in units):
            raise ValueError("evidence_units must contain EvidenceUnit objects")
        prefix = built["prompt_prefix"]
    elif "selected_evidence" in built:
        units = tuple(
            EvidenceUnit(
                source_id=str(item["record_id"]), text=item["text"],
                unit_id=str(item["sentence_index"]) if "sentence_index" in item else None,
                metadata=dict(item),
            ) for item in built["selected_evidence"]
        )
        evidence = render_evidence(units)
        if not built["prompt"].endswith(evidence):
            raise ValueError("selector prompt formatting requires an explicit evidence adapter")
        prefix = built["prompt"][:-len(evidence)] if evidence else built["prompt"]
    else:
        if len(built["contexts"]) != len(record_ids):
            raise ValueError("whole-document contexts must match record IDs")
        units = tuple(
            EvidenceUnit(str(rid), text, label=str(i))
            for i, (rid, text) in enumerate(zip(record_ids, built["contexts"]), 1)
        )
        evidence = render_evidence(units)
        if not built["prompt"].endswith(evidence):
            raise ValueError("baseline prompt formatting requires an explicit evidence adapter")
        prefix = built["prompt"][:-len(evidence)] if evidence else built["prompt"]
    if prefix + render_evidence(units) != built["prompt"]:
        raise ValueError("evidence adapter must reproduce the unbudgeted prompt exactly")
    return EvidenceBatch(units, prefix, built["prompt"])


def control_metrics(batch: EvidenceBatch, counter=None) -> dict:
    """Describe a no-budget control without packing, deduplication or changes."""
    counter = counter or TokenCounter()
    tokens = counter.count(render_evidence(batch.units))
    sources = list(dict.fromkeys(u.source_id for u in batch.units))
    return {
        "configured_budget": None, "confidence": None, "budget_reason": "disabled",
        "score_count": 0, "token_counter": counter.name, "packing_policy": "none",
        "evidence_tokens_before": tokens, "evidence_tokens_after": tokens,
        "token_reduction_pct": 0.0 if tokens else None, "budget_utilisation": None,
        "candidate_units": len(batch.units), "selected_units": len(batch.units),
        "discarded_units": 0, "truncated_units": 0,
        "duplicate_units": 0, "oversized_units": 0,
        "source_ids_before": sources, "source_ids_after": sources,
        "documents_represented": len(sources), "budget_controller_ms": 0.0,
    }


def budget_built_context(built, record_ids, scores, policy, counter=None, packing_policy="skip"):
    start = time.perf_counter()
    batch = extract_evidence(built, record_ids)
    packed = apply_budget(batch.units, scores, policy, counter, packing_policy)
    result = {
        **built, "contexts": [u.text for u in packed["units"]],
        "prompt": batch.prompt_prefix + packed["evidence"],
        "selected_evidence": [u.as_dict() for u in packed["units"]],
        "budget_metrics": packed["metrics"], "discarded_evidence": packed["discarded"],
    }
    result["budget_metrics"]["budget_controller_ms"] = (time.perf_counter() - start) * 1000
    return result


class BudgetedContextBuilder:
    """Wrap any compatible selector; a None policy is a transparent bypass.

    build_with_retrieval supplies scores for adaptive policies. The legacy build
    signature also works, with the policy's conservative missing-score fallback.
    """

    def __init__(self, base_builder, policy=None, counter=None, packing_policy="skip"):
        self.base_builder = base_builder
        self.policy = policy
        self.counter = counter or TokenCounter()
        self.packing_policy = packing_policy

    def build(self, transcript, record_ids, data):
        return self.build_with_retrieval(transcript, {"record_ids": record_ids}, data)

    def build_with_retrieval(self, transcript, retrieval, data):
        start = time.perf_counter()
        built = self.base_builder.build(transcript, retrieval["record_ids"], data)
        selector_ms = (time.perf_counter() - start) * 1000
        if self.policy is None:
            return built
        result = budget_built_context(
            built, retrieval["record_ids"], retrieval.get("scores"), self.policy,
            self.counter, self.packing_policy,
        )
        result["budget_metrics"]["selector_ms"] = selector_ms
        return result


def add_budget_arguments(parser):
    parser.add_argument("--context-budget", choices=["none", "fixed", "adaptive"], default="none")
    parser.add_argument("--evidence-token-budget", type=int, default=512)
    parser.add_argument("--adaptive-min-budget", type=int, default=256)
    parser.add_argument("--adaptive-max-budget", type=int, default=2048)
    parser.add_argument("--budget-packing-policy", choices=["skip", "sentence_prefix"], default="skip")
    parser.add_argument("--budget-tokenizer", default=None, help="Local generation-model tokenizer; no downloads")


def wrap_from_arguments(builder, args):
    if args.context_budget == "none":
        return builder
    from .core import AdaptiveBudget, FixedBudget

    policy = FixedBudget(args.evidence_token_budget) if args.context_budget == "fixed" else AdaptiveBudget(
        args.adaptive_min_budget, args.adaptive_max_budget,
    )
    counter = TokenCounter.from_local_tokenizer(args.budget_tokenizer) if args.budget_tokenizer else TokenCounter()
    return BudgetedContextBuilder(builder, policy, counter, args.budget_packing_policy)
