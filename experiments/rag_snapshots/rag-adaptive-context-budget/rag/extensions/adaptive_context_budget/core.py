"""Deterministic evidence packing and retrieval-distribution budget policies."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import re
import time
from typing import Callable, Mapping, Sequence


@dataclass(frozen=True)
class TokenCounter:
    """Count a complete rendered string, including source labels and separators.

    The dependency-free default is ceil(UTF-8 bytes / 4), an *estimate*, not the
    ELM tokenizer. The enforced bound is always against the selected counter.
    A caller can inject the actual generation tokenizer for a tokenizer bound.
    """

    name: str = "utf8_bytes_div4_estimate"
    function: Callable[[str], int] | None = field(default=None, repr=False, compare=False)

    def count(self, text: str) -> int:
        result = self.function(text) if self.function else (len(text.encode("utf-8")) + 3) // 4
        if type(result) is not int or result < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return result

    @classmethod
    def from_local_tokenizer(cls, path: str) -> TokenCounter:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        return cls(
            name=f"local_generation_tokenizer:{path}",
            function=lambda text: len(tokenizer.encode(text, add_special_tokens=False)),
        )


@dataclass(frozen=True)
class EvidenceUnit:
    source_id: str
    text: str
    unit_id: str | None = None
    label: str | None = None
    metadata: Mapping = field(default_factory=dict, compare=False)

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("evidence must have a nonempty source ID")
        if not isinstance(self.text, str):
            raise ValueError("evidence text must be a string")
        label = self.source_id if self.label is None else self.label
        if not isinstance(label, str) or any(c in label for c in "\r\n[]"):
            raise ValueError("document labels must be single-line strings without brackets")

    def render(self) -> str:
        label = self.source_id if self.label is None else self.label
        return f"[Document {label}]\n{self.text}"

    def as_dict(self) -> dict:
        return {
            **dict(self.metadata), "record_id": self.source_id, "text": self.text,
            "unit_id": self.unit_id,
            "document_label": self.source_id if self.label is None else self.label,
        }


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True)
class BudgetDecision:
    budget: int
    confidence: float | None
    reason: str
    score_count: int


@dataclass(frozen=True)
class FixedBudget:
    budget: int = 512

    def __post_init__(self):
        _nonnegative_integer(self.budget, "budget")

    def decide(self, units: Sequence[EvidenceUnit], scores=None) -> BudgetDecision:
        return BudgetDecision(self.budget, None, "fixed", 0)


@dataclass(frozen=True)
class AdaptiveBudget:
    """Use a within-query, affine-invariant top-gap statistic.

    confidence = (largest - second largest) / (largest - smallest).
    Large gaps favour small budgets; ties favour maximum evidence. At least
    three finite scores are needed: two points cannot describe score spread.
    This is a heuristic about retrieval ambiguity, not calibrated correctness.
    """

    min_budget: int = 256
    max_budget: int = 2048
    low_confidence: float = 0.1
    high_confidence: float = 0.6
    gamma: float = 1.0

    def __post_init__(self):
        _nonnegative_integer(self.min_budget, "min_budget")
        _nonnegative_integer(self.max_budget, "max_budget")
        if self.min_budget > self.max_budget:
            raise ValueError("min_budget cannot exceed max_budget")
        if not 0 <= self.low_confidence < self.high_confidence <= 1:
            raise ValueError("confidence thresholds must satisfy 0 <= low < high <= 1")
        if not math.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError("gamma must be finite and positive")

    def decide(self, units: Sequence[EvidenceUnit], scores=None) -> BudgetDecision:
        if not units:
            return BudgetDecision(self.min_budget, None, "empty_evidence", 0)
        try:
            values = [float(s) for s in scores] if scores is not None else []
        except (TypeError, ValueError):
            values = []
        if len(values) < 3 or not all(math.isfinite(s) for s in values):
            return BudgetDecision(self.max_budget, None, "missing_or_insufficient_scores", len(values))
        values.sort(reverse=True)  # Scores only; the evidence ranking is never reordered.
        scale = max(abs(s) for s in values) or 1.0
        values = [s / scale for s in values]  # Avoid overflow when subtracting finite extremes.
        spread = values[0] - values[-1]
        confidence = 0.0 if spread == 0 else (values[0] - values[1]) / spread
        fraction = (confidence - self.low_confidence) / (self.high_confidence - self.low_confidence)
        fraction = min(1.0, max(0.0, fraction)) ** self.gamma
        budget = round(self.max_budget - fraction * (self.max_budget - self.min_budget))
        return BudgetDecision(budget, confidence, "normalised_retrieval_gap", len(values))


def render_evidence(units: Sequence[EvidenceUnit]) -> str:
    return "\n\n".join(unit.render() for unit in units)


def _text_key(text):
    return " ".join(text.casefold().split())


def pack_evidence(
    units: Sequence[EvidenceUnit], budget: int, counter: TokenCounter | None = None,
    packing_policy: str = "skip",
) -> dict:
    """Keep existing order, with first-occurrence exact-text deduplication.

    'skip' continues past oversized units; 'sentence_prefix' can instead keep
    their longest fitting complete-sentence prefix. No relevance is recomputed.
    The full formatted evidence is counted on each admission, so non-additive
    tokenizers and source-label overhead cannot silently overrun the bound.
    """
    _nonnegative_integer(budget, "budget")
    if packing_policy not in ("skip", "sentence_prefix"):
        raise ValueError("unknown packing policy")
    counter = counter or TokenCounter()
    selected, discarded = [], []
    seen, kept_text = set(), set()
    evidence = ""
    truncated_count = 0
    for index, unit in enumerate(units):
        key = _text_key(unit.text)
        reason = "empty" if not key else "duplicate" if key in seen else None
        seen.add(key)
        if reason:
            discarded.append({"index": index, "record_id": unit.source_id, "reason": reason})
            continue
        prefix = evidence + ("\n\n" if selected else "")
        chosen = unit
        if counter.count(prefix + unit.render()) > budget:
            chosen = None
            if packing_policy == "sentence_prefix":
                # This is a prefix packing rule, not sentence ranking/selection.
                for match in re.finditer(r'''[.!?](?:["')\]]*)(?=\s|$)''', unit.text):
                    candidate = replace(unit, text=unit.text[:match.end()].rstrip())
                    if counter.count(prefix + candidate.render()) <= budget:
                        chosen = candidate
            if chosen is None:
                discarded.append({"index": index, "record_id": unit.source_id, "reason": "over_budget"})
                continue
        if _text_key(chosen.text) in kept_text:
            discarded.append({"index": index, "record_id": unit.source_id, "reason": "duplicate_prefix"})
            continue
        # Recheck the complete string, even for custom/non-additive counters.
        trial = prefix + chosen.render()
        if counter.count(trial) > budget:
            raise ValueError("token counter changed during deterministic packing")
        selected.append(chosen)
        kept_text.add(_text_key(chosen.text))
        evidence = trial
        truncated_count += chosen.text != unit.text
    return {
        "units": selected, "evidence": evidence, "tokens": counter.count(evidence),
        "discarded": discarded, "truncated_units": truncated_count,
    }


def apply_budget(units, scores, policy, counter=None, packing_policy="skip") -> dict:
    start = time.perf_counter()
    counter = counter or TokenCounter()
    decision = policy.decide(units, scores)
    before = counter.count(render_evidence(units))
    packed = pack_evidence(units, decision.budget, counter, packing_policy)
    selected = packed["units"]
    metrics = {
        "configured_budget": decision.budget, "confidence": decision.confidence,
        "budget_reason": decision.reason, "score_count": decision.score_count,
        "token_counter": counter.name, "packing_policy": packing_policy,
        "evidence_tokens_before": before, "evidence_tokens_after": packed["tokens"],
        "token_reduction_pct": 100 * (1 - packed["tokens"] / before) if before else None,
        "budget_utilisation": packed["tokens"] / decision.budget if decision.budget else None,
        "candidate_units": len(units), "selected_units": len(selected),
        "discarded_units": len(units) - len(selected),
        "duplicate_units": sum(d["reason"] in ("duplicate", "duplicate_prefix") for d in packed["discarded"]),
        "oversized_units": sum(d["reason"] == "over_budget" for d in packed["discarded"]),
        "truncated_units": packed["truncated_units"],
        "source_ids_before": list(dict.fromkeys(u.source_id for u in units)),
        "source_ids_after": list(dict.fromkeys(u.source_id for u in selected)),
        "documents_represented": len({u.source_id for u in selected}),
    }
    metrics["budget_controller_ms"] = (time.perf_counter() - start) * 1000
    return {**packed, "metrics": metrics}
