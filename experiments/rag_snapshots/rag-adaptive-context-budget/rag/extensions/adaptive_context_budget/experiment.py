"""Freeze retrieval/selection once, then compare only evidence budget policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import time

from .core import AdaptiveBudget, FixedBudget, TokenCounter
from .integration import budget_built_context, control_metrics, extract_evidence


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    from src.evaluation.answer_judge import load_jsonl
    return load_jsonl(Path(path))


def write_json(path, value):
    from src.evaluation.answer_judge import write_json_atomic
    write_json_atomic(Path(path), value)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def make_builders(selectors, max_chars=5000, mmr_lambda=0.7):
    from src.context_builder import SimpleContextBuilder

    builders = {}
    for selector in selectors:
        if selector == "simple":
            builders[selector] = SimpleContextBuilder()
        elif selector in ("tfidf", "mmr"):
            try:
                from src.context_builder import ExperimentalContextBuilder
            except ImportError as exc:
                raise RuntimeError("Integrate the teammate context-selection branch first") from exc
            builders[selector] = ExperimentalContextBuilder(
                strategy=selector, max_chars=max_chars, mmr_lambda=mmr_lambda,
            )
        else:
            raise ValueError(f"unknown selector: {selector}")
    return builders


def prepare_query(item, transcript, data, encoder, retriever, builders, k=5, reranker=None, final_k=5):
    """No qrels/answers enter this path. ASR transcripts must already be cached."""
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("a nonempty cached ASR transcript is required")
    start = time.perf_counter()
    vector = encoder.encode(transcript)
    embedding_ms = (time.perf_counter() - start) * 1000
    retrieval = retriever.retrieve(vector, k=k)
    final = retrieval
    rerank_ms = 0.0
    if reranker is not None:
        start = time.perf_counter()
        final = reranker.rerank(transcript, retrieval, data, final_k=final_k)
        rerank_ms = (time.perf_counter() - start) * 1000
    selections = {}
    for name, builder in builders.items():
        start = time.perf_counter()
        built = builder.build(transcript, final["record_ids"], data)
        elapsed = (time.perf_counter() - start) * 1000
        extract_evidence(built, final["record_ids"])  # Verify a lossless adapter before API calls.
        selections[name] = {"built": built, "selector_ms": elapsed, "evidence_hash": stable_hash(built)}
    row = {
        "query_id": item["query_id"], "split": item["split"],
        "audio_file": item.get("audio_file"), "transcript": transcript,
        "retrieval": retrieval, "final_retrieval": final, "selections": selections,
        "preparation_timing": {"embedding_ms": embedding_ms, "retrieval_ms": retrieval["latency_ms"],
                               "rerank_ms": rerank_ms},
    }
    row["candidate_ranking_hash"] = stable_hash({
        "transcript": transcript, "record_ids": final["record_ids"], "scores": final.get("scores"),
    })
    return row


@dataclass(frozen=True)
class Condition:
    name: str
    selector: str
    policy: FixedBudget | AdaptiveBudget | None = None
    packing_policy: str = "skip"
    matched_target_tokens: float | None = None


def render_condition(query, condition, counter):
    selection = query["selections"][condition.selector]
    built = selection["built"]
    ids = query["final_retrieval"]["record_ids"]
    if condition.policy is None:
        batch = extract_evidence(built, ids)
        return built, control_metrics(batch, counter)
    result = budget_built_context(
        built, ids, query["final_retrieval"].get("scores"), condition.policy,
        counter, condition.packing_policy,
    )
    return result, result["budget_metrics"]


def match_fixed_budget(queries, adaptive, counter, max_budget):
    """Match mean *used* evidence tokens using inputs only, never judge scores."""
    if not queries:
        raise ValueError("cannot match an empty query set")
    target = sum(render_condition(q, adaptive, counter)[1]["evidence_tokens_after"] for q in queries) / len(queries)
    cache = {}

    def used(budget):
        if budget not in cache:
            spec = Condition("calibration", adaptive.selector, FixedBudget(budget), adaptive.packing_policy)
            cache[budget] = sum(render_condition(q, spec, counter)[1]["evidence_tokens_after"] for q in queries) / len(queries)
        return cache[budget]

    low, high = 0, max_budget
    while low <= high:
        middle = (low + high) // 2
        if used(middle) < target:
            low = middle + 1
        else:
            high = middle - 1
    for budget in {0, max_budget, max(0, min(low, max_budget)), max(0, min(high, max_budget))}:
        used(budget)
    best = min(cache, key=lambda budget: (abs(cache[budget] - target), budget))
    return best, target, cache[best]


def make_conditions(queries, budgets=(256, 512, 1024, 2048), packing_policies=("skip", "sentence_prefix"),
                    adaptive=None, counter=None, match_adaptive=False):
    if not queries:
        raise ValueError("snapshot must contain queries")
    selectors = list(queries[0]["selections"])
    if any(set(q["selections"]) != set(selectors) for q in queries):
        raise ValueError("every query must have the same selectors")
    adaptive = adaptive or AdaptiveBudget()
    counter = counter or TokenCounter()
    conditions = []
    for selector in selectors:
        conditions.append(Condition(f"{selector}_none", selector))
        for packing in packing_policies:
            for budget in budgets:
                conditions.append(Condition(f"{selector}_fixed_{budget}_{packing}", selector, FixedBudget(budget), packing))
            spec = Condition(f"{selector}_adaptive_{packing}", selector, adaptive, packing)
            conditions.append(spec)
            if match_adaptive:
                budget, target, _ = match_fixed_budget(queries, spec, counter, adaptive.max_budget)
                conditions.append(Condition(f"{selector}_matched_{packing}", selector, FixedBudget(budget), packing, target))
    names = [spec.name for spec in conditions]
    if len(names) != len(set(names)):
        raise ValueError("duplicate condition names; remove repeated budgets or policies")
    return conditions


def validate_snapshot(queries):
    ids = [q["query_id"] for q in queries]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("snapshot query IDs must be nonempty and unique")
    for query in queries:
        if query["split"] not in ("practice", "dev") or not query["transcript"].strip():
            raise ValueError("snapshot requires a split and nonempty ASR transcript")
        expected = stable_hash({"transcript": query["transcript"],
                                "record_ids": query["final_retrieval"]["record_ids"],
                                "scores": query["final_retrieval"].get("scores")})
        if query["candidate_ranking_hash"] != expected:
            raise ValueError("snapshot ranking/transcript was modified")
        for selection in query["selections"].values():
            if selection["evidence_hash"] != stable_hash(selection["built"]):
                raise ValueError("snapshot selector output was modified")


def evaluate_condition(query, condition, counter, generator=None, qrels=None):
    from src.generator import SYSTEM_PROMPT

    start = time.perf_counter()
    built, metrics = render_condition(query, condition, counter)
    metrics = dict(metrics)
    selector_ms = query["selections"][condition.selector]["selector_ms"]
    generation, error = None, None
    if generator is not None:
        try:
            generation = generator.generate(built["prompt"])
        except Exception as exc:
            error = type(exc).__name__  # Avoid storing credentials/request headers in results.
    wall_ms = (time.perf_counter() - start) * 1000
    usage = (generation or {}).get("usage") or {}
    generation_ms = (generation or {}).get("latency_ms")
    timing = {
        **query["preparation_timing"], "selector_ms": selector_ms,
        "budget_controller_ms": metrics["budget_controller_ms"],
        "context_builder_ms": selector_ms + metrics["budget_controller_ms"],
        "generation_ms": generation_ms, "budget_and_generation_wall_ms": wall_ms,
        "end_to_end_ms": None,  # ASR is cached; do not invent an end-to-end measurement.
    }
    shared_ms = sum(query["preparation_timing"].values())
    timing["reconstructed_downstream_ms"] = (
        shared_ms + timing["context_builder_ms"] + generation_ms if generation_ms is not None else None
    )
    # Common analysis strata must also exist for no-budget/fixed controls.
    original_batch = extract_evidence(query["selections"][condition.selector]["built"],
                                      query["final_retrieval"]["record_ids"])
    query_confidence = AdaptiveBudget().decide(
        original_batch.units, query["final_retrieval"].get("scores"),
    ).confidence
    row = {
        "query_id": query["query_id"], "split": query["split"], "audio_file": query.get("audio_file"),
        "transcript": query["transcript"], "condition": condition.name, "selector": condition.selector,
        "candidate_ranking_hash": query["candidate_ranking_hash"],
        "selector_evidence_hash": query["selections"][condition.selector]["evidence_hash"],
        "query_retrieval_confidence": query_confidence,
        "retrieved_record_ids": query["retrieval"]["record_ids"],
        "final_retrieved_record_ids": query["final_retrieval"]["record_ids"],
        "prompt": built["prompt"], "contexts": built["contexts"],
        "selected_evidence": built.get("selected_evidence", []),
        "discarded_evidence": built.get("discarded_evidence", []),
        "estimated_prompt_text_tokens": counter.count(built["prompt"]) + counter.count(SYSTEM_PROMPT),
        "prompt_estimate_excludes_chat_template": True,
        "api_input_tokens": usage.get("prompt_tokens"), "api_output_tokens": usage.get("completion_tokens"),
        "api_total_tokens": usage.get("total_tokens"), "generation_model": (generation or {}).get("model"),
        "generation_status": "not_requested" if generator is None else "failed" if error else "generated",
        "generation_error": error, **metrics, "timing": timing,
    }
    if generation is not None:
        row["answer"] = generation["answer"]
    if qrels is not None:
        from src.metrics import recall_at_k

        relevant = qrels.get(query["query_id"], {})
        before = set(metrics["source_ids_before"]) & set(relevant)
        after = set(metrics["source_ids_after"]) & set(relevant)
        row["relevant_documents_before"] = len(before)
        row["relevant_documents_after"] = len(after)
        row["relevant_document_retention"] = len(after) / len(before) if before else None
        row["relevant_document_fraction"] = (
            recall_at_k(metrics["source_ids_after"], relevant, len(metrics["source_ids_after"])) if relevant else None
        )
    return row


def run_comparison(queries, conditions, output, counter=None, generator=None, qrels=None, provenance=None):
    validate_snapshot(queries)
    if not conditions or len({c.name for c in conditions}) != len(conditions):
        raise ValueError("provide at least one uniquely named condition")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    counter = counter or TokenCounter()
    write_json(output / "run_manifest.json", {
        "query_ids": [q["query_id"] for q in queries], "snapshot_hash": stable_hash(queries),
        "conditions": [asdict(c) for c in conditions], "token_counter": counter.name,
        "generation_requested": generator is not None,
        "provenance": provenance or {},
        "generation_model": getattr(generator, "model", None), "generation_temperature": 0.1,
        "timing_note": "Shared ASR/retrieval/selection cached; reconstructed downstream time is not wall-clock E2E",
        "matched_budget_note": "Matched on this snapshot's used evidence tokens, without answers/qrels; discrete fit may be imperfect",
    })
    streams, candidates = {}, {}
    all_rows = {c.name: [] for c in conditions}
    try:
        for condition in conditions:
            folder = output / condition.name
            folder.mkdir()
            streams[condition.name] = (folder / "metrics.jsonl").open("x", encoding="utf-8")
            if generator is not None:
                candidates[condition.name] = (folder / "candidates.jsonl").open("x", encoding="utf-8")
        for index, query in enumerate(queries):
            # Rotate condition order across queries to avoid always calling one model condition first.
            offset = index % len(conditions)
            order = conditions[offset:] + conditions[:offset]
            for condition in order:
                row = evaluate_condition(query, condition, counter, generator, qrels)
                all_rows[condition.name].append(row)
                stream = streams[condition.name]
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                if "answer" in row:
                    candidates[condition.name].write(json.dumps(row, ensure_ascii=False) + "\n")
                    candidates[condition.name].flush()
        for condition in conditions:
            rows = all_rows[condition.name]
            flattened = [{**{k: v for k, v in r.items() if k not in ("prompt", "contexts", "selected_evidence", "timing")},
                          **r["timing"]} for r in rows]
            write_csv(output / condition.name / "metrics.csv", flattened)
    finally:
        for stream in list(streams.values()) + list(candidates.values()):
            stream.close()
    return all_rows
