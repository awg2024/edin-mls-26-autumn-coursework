"""Controlled rebuild/incremental comparisons with an independent logical oracle."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

import numpy as np

from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.retriever import ExactDenseRetriever
from .index import CompactionPolicy, DynamicCorpus, FullRebuildCorpus
from .io import fingerprint, write_csv, write_json_atomic, write_jsonl_atomic
from .workload import apply_oracle, make_mutation_log, materialise_oracle, mutation_from_entry


@dataclass(frozen=True)
class Case:
    family: str = "principal"
    corpus_size: int = 2000
    operations: int = 200
    batch_size: int = 1
    profile: str = "mixed"
    queries_per_update: int = 1
    compaction_threshold: float = 0.05
    checkpoint_every: int = 50
    initial_fraction: float = 0.8
    update_mode: str = "refresh"
    seed: int = 2026

    def __post_init__(self):
        if self.operations < 1 or self.batch_size < 1 or self.queries_per_update < 0 or self.checkpoint_every < 1:
            raise ValueError("invalid workload sizes")
        CompactionPolicy(self.compaction_threshold, self.compaction_threshold)

    @property
    def name(self):
        return self.family + "_" + fingerprint(asdict(self))[:10]


def suite_cases(base_size=2000, corpus_sizes=(2000, 5000, 10000, 24000), seed=2026):
    base = Case(corpus_size=base_size, seed=seed)
    result = [base]
    result.extend(replace(base, family="batch_size", operations=500, batch_size=n, checkpoint_every=100)
                  for n in (1, 10, 100, 500))
    result.extend(replace(base, family="mutation_type", operations=100, profile=kind)
                  for kind in ("insert", "update", "delete", "mixed"))
    result.append(replace(base, family="accumulation", operations=300, batch_size=10, checkpoint_every=10))
    result.extend(replace(base, family="threshold", operations=300, batch_size=10,
                          checkpoint_every=50, compaction_threshold=threshold)
                  for threshold in (0.01, 0.05, 0.1, 0.2))
    result.extend(replace(base, family="query_ratio", operations=10, checkpoint_every=10,
                          queries_per_update=ratio) for ratio in (1, 10, 100))
    result.extend(replace(base, family="corpus_size", corpus_size=n, operations=100, batch_size=10)
                  for n in corpus_sizes)
    return result


def _reference(oracle, dimension):
    if oracle:
        ids, vectors, records = materialise_oracle(oracle)
    else:
        ids, vectors, records = [], np.empty((0, dimension), np.float32), {}
    retriever = ExactDenseRetriever.from_arrays(vectors, ids, deterministic_ties=True, reproducible=True)
    return retriever, records, vectors, ids


def validate_checkpoint(index, oracle, sources, catalog, queries, k, qrels=None):
    reference, records, vectors, ids = _reference(oracle, index.dimension)
    snapshot = index.snapshot()
    got_records, got_vectors, got_ids = snapshot.export()
    logical_equal = got_records == records and got_ids == ids and np.array_equal(got_vectors, vectors)
    rows = []
    semantic_replacements = any(rid != source for rid, source in sources.items())
    for qid, vector in zip(queries.ids, queries.vectors):
        expected, got = reference.retrieve(vector, k), snapshot.retrieve(vector, k)
        same_ids = expected["record_ids"] == got["record_ids"]
        scores_equal = len(expected["scores"]) == len(got["scores"]) and np.allclose(
            expected["scores"], got["scores"], atol=2e-6, rtol=2e-6)
        max_error = max((abs(a - b) for a, b in zip(expected["scores"], got["scores"])), default=0.0)
        current = all(rid in oracle and snapshot.record_text(rid) == oracle[rid][0] for rid in got["record_ids"])
        row = {
            "query_id": qid, "expected_ids": expected["record_ids"], "returned_ids": got["record_ids"],
            "expected_scores": expected["scores"], "returned_scores": got["scores"],
            "exact_topk_agreement": same_ids, "scores_within_tolerance": bool(scores_equal),
            "max_score_error": max_error, "all_active_and_current_text": current,
            "logical_state_equal": logical_equal, "corpus_version": snapshot.version,
            "original_qrels_recall_at10": None, "active_qrels_recall_at10": None,
            "active_qrels_mrr_at10": None, "active_qrels_ndcg_at10": None,
            "relevant_document_availability": None,
        }
        relevant = qrels.get(qid, {}) if qrels is not None else {}
        if relevant and not semantic_replacements:
            active_relevant = {rid: value for rid, value in relevant.items() if rid in oracle}
            row["relevant_document_availability"] = len(active_relevant) / len(relevant)
            row["original_qrels_recall_at10"] = recall_at_k(got["record_ids"], relevant, 10)
            if active_relevant:
                row.update(active_qrels_recall_at10=recall_at_k(got["record_ids"], active_relevant, 10),
                           active_qrels_mrr_at10=mrr_at_k(got["record_ids"], active_relevant, 10),
                           active_qrels_ndcg_at10=ndcg_at_k(got["record_ids"], active_relevant, 10))
            row["quality_scope"] = "Original denominator plus separately labelled active-original-text scope"
        else:
            row["quality_scope"] = ("semantic_replacements_invalidate_original_qrels" if semantic_replacements else
                                    "no_judged_query_vectors_or_qrels; correctness_only")
        rows.append(row)
    return rows


def _new_index(strategy, records, matrix, ids, catalog, threshold):
    kwargs = {"embedding_space": catalog.embedding_space}
    if strategy == "full_rebuild":
        return FullRebuildCorpus(records, matrix, ids, **kwargs)
    if strategy == "incremental_compact":
        kwargs["compaction"] = CompactionPolicy(threshold, threshold)
    return DynamicCorpus(records, matrix, ids, **kwargs)


def _stats(values):
    if not values:
        return None, None, None
    ordered = sorted(values)
    return statistics.mean(values), statistics.median(values), ordered[max(0, math.ceil(0.95 * len(values)) - 1)]


def run_case(catalog, queries, case, output, repeats=3, k=10, qrels=None, trace_memory=True):
    if repeats < 1 or k < 1:
        raise ValueError("repeats and k must be positive")
    if qrels is not None and k < 10:
        raise ValueError("use a fixed k>=10 to report the requested retrieval metrics")
    if queries.embedding_space != catalog.embedding_space or queries.vectors.shape[1] != catalog.vectors.shape[1]:
        raise ValueError("query/catalog embedding spaces or dimensions differ")
    initial_ids = catalog.initial_ids(case.initial_fraction, case.seed)
    initial_records = {rid: catalog.records[rid] for rid in initial_ids}
    initial_matrix = np.stack([catalog.vector(rid) for rid in initial_ids])
    log = make_mutation_log(catalog, initial_ids, case.operations, case.profile, case.update_mode, case.seed)
    batches = [log[start:start + case.batch_size] for start in range(0, len(log), case.batch_size)]
    folder = Path(output) / case.name
    folder.mkdir(parents=True, exist_ok=False)
    write_json_atomic(folder/"config.json", {
        **asdict(case), "dataset": catalog.source, "embedding_space": catalog.embedding_space,
        "dimension": catalog.vectors.shape[1], "initial_active_records": len(initial_ids),
        "initial_ids": initial_ids, "catalog_fingerprint": catalog.fingerprint(),
        "query_source": queries.source, "query_ids": queries.ids, "k": k, "repeats": repeats,
        "actual_mutation_counts": dict(Counter(e["kind"] for e in log)),
        "mutation_log_hash": fingerprint(log),
        "tie_policy": "exact FP32 row scores descending, then record ID ascending; no score quantisation",
        "compaction_rule": "physical delta rows / active >= threshold OR base tombstone fraction >= threshold",
        "timing": "Operation/query wall times exclude oracle validation, result serialization and harness work",
    })
    write_jsonl_atomic(folder/"mutations.jsonl", log)
    np.savez_compressed(folder/"queries.npz", vectors=queries.vectors, query_ids=np.asarray(queries.ids))
    summaries, mutation_rows, operation_rows, query_rows, correctness_rows, history = [], [], [], [], [], []
    strategies = ["full_rebuild", "incremental", "incremental_compact"]

    for repeat in range(repeats):
        order = strategies[repeat % 3:] + strategies[:repeat % 3]
        for strategy in order:
            tag = {"case": case.name, "family": case.family, "strategy": strategy, "repeat": repeat}
            start = time.perf_counter()
            index = _new_index(strategy, initial_records, initial_matrix, initial_ids, catalog, case.compaction_threshold)
            initial_build_ms = (time.perf_counter() - start) * 1000
            oracle = {rid: (catalog.records[rid], catalog.vector(rid)) for rid in initial_ids}
            sources = {rid: rid for rid in initial_ids}
            local_queries, local_batches, type_latency = [], [], {kind: [] for kind in ("insert", "update", "delete")}
            query_cursor = 0

            def timed_queries(number, processed, phase, in_workload=False):
                nonlocal query_cursor
                timings = []
                for query_number in range(number):
                    position = (query_cursor if in_workload else query_number) % len(queries.ids)
                    if in_workload:
                        query_cursor += 1
                    start = time.perf_counter()
                    result = index.search(queries.vectors[position], k)
                    elapsed = (time.perf_counter() - start) * 1000
                    row = {**tag, "processed_operations": processed, "phase": phase,
                           "in_workload": in_workload, "query_id": queries.ids[position],
                           "query_wall_ms": elapsed, "retriever_reported_ms": result["latency_ms"],
                           "record_ids": result["record_ids"], "scores": result["scores"],
                           "corpus_version": result["corpus_version"]}
                    local_queries.append(row)
                    query_rows.append(row)
                    timings.append(elapsed)
                return timings

            def checkpoint(processed, phase):
                checks = validate_checkpoint(index, oracle, sources, catalog, queries, k, qrels)
                correctness_rows.extend({**tag, "processed_operations": processed, "phase": phase, **r} for r in checks)
                if any(not (r["exact_topk_agreement"] and r["scores_within_tolerance"] and
                            r["all_active_and_current_text"] and r["logical_state_equal"]) for r in checks):
                    write_json_atomic(folder/"correctness_failure.json", {**tag, "processed": processed, "checks": checks})
                    raise AssertionError("exact retrieval mismatch; see correctness_failure.json")
                timings = timed_queries(len(queries.ids), processed, phase)
                stats = index.get_stats()
                history.append({**stats, **tag, "processed_operations": processed, "phase": phase,
                                "probe_query_mean_ms": statistics.mean(timings)})

            # Equal warmups for all strategies, separate from recorded diagnostic queries.
            for vector in queries.vectors:
                index.search(vector, k)
            checkpoint(0, "before_updates")
            processed = 0
            for batch_number, entries in enumerate(batches):
                operations = [mutation_from_entry(catalog, entry) for entry in entries]
                start = time.perf_counter()
                receipt = index.apply_batch(operations, trace_operations=True)
                elapsed = (time.perf_counter() - start) * 1000
                processed += len(entries)
                row = {**tag, "batch": batch_number, "processed_operations": processed,
                       "batch_size": len(entries), "mutation_wall_ms": elapsed,
                       "amortised_ms_per_operation": elapsed / len(entries),
                       **{key: value for key, value in receipt.items() if key != "operation_timings"}}
                mutation_rows.append(row)
                local_batches.append(row)
                for entry, trace in zip(entries, receipt["operation_timings"]):
                    operation_rows.append({**tag, "batch": batch_number, **entry,
                                           "staging_ms": trace["staging_ms"],
                                           "observed_single_operation_wall_ms": elapsed if len(entries) == 1 else None,
                                           "amortised_batch_ms_per_operation": elapsed / len(entries)})
                    if len(entries) == 1:
                        type_latency[entry["kind"]].append(elapsed)
                    apply_oracle(oracle, sources, catalog, entry)
                timed_queries(len(entries) * case.queries_per_update, processed, "after_batch", True)
                if processed % case.checkpoint_every == 0 or processed == len(log):
                    checkpoint(processed, "checkpoint")
            pre_compaction = index.get_stats()
            manual = index.compact()
            checkpoint(len(log), "after_manual_compaction")
            workload = [r["query_wall_ms"] for r in local_queries if r["in_workload"]]
            before = [r["query_wall_ms"] for r in local_queries if r["phase"] == "before_updates"]
            after = [r["query_wall_ms"] for r in local_queries if r["phase"] == "after_manual_compaction"]
            qmean, qmedian, qp95 = _stats(workload)
            mutation_ms = sum(r["mutation_wall_ms"] for r in local_batches)
            summary = {
                **tag, **asdict(case), "dataset": catalog.source, "dimension": catalog.vectors.shape[1],
                "initial_active_records": len(initial_ids), "initial_build_ms": initial_build_ms,
                "insert_wall_mean_ms": statistics.mean(type_latency["insert"]) if type_latency["insert"] else None,
                "update_wall_mean_ms": statistics.mean(type_latency["update"]) if type_latency["update"] else None,
                "delete_wall_mean_ms": statistics.mean(type_latency["delete"]) if type_latency["delete"] else None,
                "batch_wall_mean_ms": statistics.mean(r["mutation_wall_ms"] for r in local_batches),
                "amortised_update_ms": mutation_ms / len(log),
                "updates_per_second": len(log) / (mutation_ms / 1000),
                "total_mutation_ms": mutation_ms,
                "total_staging_ms": sum(r["mutation_ms"] for r in local_batches),
                "total_rebuild_ms": sum(r["rebuild_ms"] for r in local_batches),
                "total_auto_compaction_ms": sum(r["compaction_ms"] for r in local_batches),
                "automatic_compactions": sum(r["compaction_reason"] is not None and strategy != "full_rebuild" for r in local_batches),
                "query_mean_ms": qmean, "query_median_ms": qmedian, "query_p95_ms": qp95,
                "workload_queries": len(workload), "query_time_sum_ms": sum(workload),
                "search_only_queries_per_second": len(workload) / (sum(workload)/1000) if workload else None,
                "query_before_mean_ms": statistics.mean(before), "query_after_manual_compaction_mean_ms": statistics.mean(after),
                "measured_workload_ms": mutation_ms + sum(workload),
                "manual_compaction_ms": manual["compaction_ms"],
                "embedding_bytes_before_manual_compaction": pre_compaction["embedding_bytes"],
                "embedding_bytes_after_manual_compaction": index.get_stats()["embedding_bytes"],
                "delta_rows_before_manual_compaction": pre_compaction["delta_rows"],
                "tombstones_before_manual_compaction": pre_compaction["tombstoned_base_rows"],
                "exact_topk_agreement": 1.0, "correctness": "PASS",
                "maintenance_traced_peak_bytes": None,
            }
            # Memory instrumentation is a separate replay, never included in latency samples.
            if trace_memory and repeat == 0:
                probe = _new_index(strategy, initial_records, initial_matrix, initial_ids, catalog, case.compaction_threshold)
                tracemalloc.start()
                for entries in batches:
                    probe.apply_batch([mutation_from_entry(catalog, entry) for entry in entries])
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                summary["maintenance_traced_peak_bytes"] = peak
            summaries.append(summary)
            print(f"{case.family}: N={len(catalog.ids)}, batch={case.batch_size}, {strategy}, repeat={repeat+1}: PASS", flush=True)
    write_csv(folder/"summary_per_repeat.csv", summaries)
    write_csv(folder/"batch_timings.csv", mutation_rows)
    write_csv(folder/"operation_timings.csv", operation_rows)
    write_csv(folder/"query_timings.csv", query_rows)
    write_csv(folder/"retrieval_checks.csv", correctness_rows)
    write_csv(folder/"state_history.csv", history)
    return summaries


def run_suite(catalog, queries, cases, output, repeats=3, k=10, qrels=None, trace_memory=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output/"experiment_manifest.json", {
        "dataset": catalog.source, "embedding_space": catalog.embedding_space, "source_records": len(catalog.ids),
        "source_catalog_fingerprint": catalog.fingerprint(), "query_source": queries.source,
        "platform": platform.platform(), "python": sys.version, "numpy": np.__version__,
        "repeats": repeats, "k": k, "cases": [asdict(case) for case in cases],
        "strategy_order": "rotated by repetition", "query_warmup": "all probe vectors once before updates",
        "score_mode": "shared exact FP32 row reduction; ascending IDs for exact score ties",
        "workload_time": "sum of operation and scheduled query wall times; validation/IO/initial build/manual diagnostic compaction excluded",
        "memory": "current-state embedding bytes exact; separate first-repeat tracemalloc additional peak, not RSS",
        "qrels_present": qrels is not None,
    })
    all_rows = []
    for case in cases:
        subset = catalog.subset(case.corpus_size, case.seed)
        all_rows.extend(run_case(subset, queries, case, output, repeats, k, qrels, trace_memory))
        write_csv(output/"all_summary_per_repeat.csv", all_rows)
    from .plots import summarise
    summarise(output)
    return all_rows
