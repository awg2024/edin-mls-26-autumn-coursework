"""Small matched trial using cached *real* ASR/encoder outputs and the supplied ELM.

Identical prompts reuse one generated answer to avoid redundant remote requests.
This is evidence equivalence, not an independent model reproducibility experiment.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever
from .index import DynamicCorpus, FullRebuildCorpus
from .io import load_jsonl, write_csv, write_json_atomic, write_jsonl_atomic
from .pipeline import DynamicRAGPipeline
from .workload import make_mutation_log, mutation_from_entry


class _CachedASR:
    def __init__(self, queries):
        self.values = dict(zip(queries.ids, queries.transcripts))

    def transcribe(self, audio_path):
        return self.values[str(audio_path)]


class _CachedEncoder:
    def __init__(self, queries):
        self.values = {}
        for text, vector in zip(queries.transcripts, queries.vectors):
            if text in self.values and not (self.values[text] == vector).all():
                raise ValueError("same transcript has different cached embeddings")
            self.values[text] = vector

    def encode(self, text):
        return self.values[text]


class _GenerationCache:
    def __init__(self, generator):
        self.generator, self.cache, self.last = generator, {}, {}

    def generate(self, prompt):
        hit = prompt in self.cache
        if not hit:
            self.cache[prompt] = self.generator.generate(prompt)
        original = self.cache[prompt]
        self.last = {"generation_cache_hit": hit, "original_generation_ms": original["latency_ms"]}
        return {**original, "latency_ms": 0.0 if hit else original["latency_ms"]}


def run_trial(catalog, queries, output, generator, builder, *, operations=20, seed=2026,
              k=10, final_k=5, reranker=None):
    if not queries.transcripts or not queries.split or "vector_probes" in queries.source:
        raise ValueError("real transcript/query-encoder cache required; vector probes cannot generate a RAG trial")
    if queries.embedding_space != catalog.embedding_space or queries.vectors.shape[1] != catalog.vectors.shape[1]:
        raise ValueError("query/catalog space or dimension mismatch")
    if not queries.source.startswith("real_asr_") or not catalog.source.startswith("course_precomputed_"):
        raise ValueError("rag-trial requires real course documents and ASR transcripts")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ids = catalog.initial_ids(0.8, seed)
    records = {rid: catalog.records[rid] for rid in ids}
    matrix = catalog.vectors[[catalog.positions[rid] for rid in ids]]
    before = DynamicCorpus(records, matrix, ids, embedding_space=catalog.embedding_space)
    after = DynamicCorpus(records, matrix, ids, embedding_space=catalog.embedding_space)
    log = make_mutation_log(catalog, ids, operations, "mixed", "refresh", seed)
    after.apply_batch([mutation_from_entry(catalog, entry) for entry in log])
    updated_records, updated_matrix, updated_ids = after.snapshot().export()
    rebuilt = FullRebuildCorpus(updated_records, updated_matrix, updated_ids, embedding_space=catalog.embedding_space)
    write_jsonl_atomic(output/"mutations.jsonl", log)
    write_json_atomic(output/"config.json", {
        "catalog_fingerprint": catalog.fingerprint(), "seed": seed, "operations": operations,
        "k": k, "final_k": final_k, "query_ids": queries.ids, "query_source": queries.source,
        "initial_fraction": 0.8, "update_mode": "refresh; original text/vector pairs",
        "generator_model": getattr(generator, "model", type(generator).__name__),
        "latency_scope": "Cached real ASR + cached real query encoding; downstream pipeline wall time, not live audio E2E",
        "generation": "One real generation per unique prompt, shared across matched conditions; cache hits have zero generation_ms",
        "quality_scope": "Supplied evaluator uses original gold; before/after differ in document availability",
    })
    shared_generator = _GenerationCache(generator)
    common = dict(asr=_CachedASR(queries), encoder=_CachedEncoder(queries), context_builder=builder,
                  generator=shared_generator, reranker=reranker)
    data = SimpleNamespace(record_text=lambda rid: catalog.records[rid])
    static = RAGPipeline(retriever=ExactDenseRetriever.from_arrays(catalog.vectors, catalog.ids), data=data, **common)
    pipelines = {"original_static": static,
                 "dynamic_before": DynamicRAGPipeline(corpus=before, **common),
                 "dynamic_after": DynamicRAGPipeline(corpus=after, **common),
                 "freshly_rebuilt_after": DynamicRAGPipeline(corpus=rebuilt, **common)}
    rows_by_condition = {}
    for condition, pipeline in pipelines.items():
        rows = []
        for qid in queries.ids:
            result = pipeline.answer_audio(qid, k=k, final_k=final_k)
            rows.append({"query_id": qid, "split": queries.split, "condition": condition,
                         "latency_scope": "cached_asr_and_query_encoding", **result, **shared_generator.last})
            # Save after each real call so an interrupted API run retains completed answers.
            write_jsonl_atomic(output/f"{condition}.jsonl", rows)
        rows_by_condition[condition] = rows
    checks = []
    for dynamic, rebuilt_row in zip(rows_by_condition["dynamic_after"], rows_by_condition["freshly_rebuilt_after"]):
        same = all(dynamic[key] == rebuilt_row[key] for key in ("prompt", "contexts", "selected_evidence", "answer"))
        same = same and dynamic["retrieval"]["record_ids"] == rebuilt_row["retrieval"]["record_ids"]
        same = same and np.allclose(dynamic["retrieval"]["scores"], rebuilt_row["retrieval"]["scores"], atol=2e-6, rtol=2e-6)
        checks.append({"query_id": dynamic["query_id"], "same_evidence_prompt_cached_answer": same})
    write_json_atomic(output/"matched_checks.json", checks)
    if not all(row["same_evidence_prompt_cached_answer"] for row in checks):
        raise AssertionError("dynamic/rebuilt evidence mismatch")
    collect_trial_scores(output)
    return rows_by_condition


def collect_trial_scores(output):
    """Join supplied evaluator outputs; absent judgements remain empty, never zero."""
    output = Path(output)
    combined = []
    for condition in ("original_static", "dynamic_before", "dynamic_after", "freshly_rebuilt_after"):
        judge_path = output/f"judge_{condition}"/"answer_judge_per_query.jsonl"
        judged = {row["query_id"]: row for row in load_jsonl(judge_path)} if judge_path.exists() else {}
        for row in load_jsonl(output/f"{condition}.jsonl"):
            score = judged.get(row["query_id"], {})
            combined.append({
                "condition": condition, "query_id": row["query_id"], "transcript": row["transcript"],
                "record_ids": row["retrieval"]["record_ids"], "final_record_ids": row["final_retrieval"]["record_ids"],
                "selected_evidence": row["selected_evidence"], "contexts": row["contexts"], "answer": row["answer"],
                "corpus_version": row.get("corpus_version"), "retrieval_ms": row["timing"]["retrieval_ms"],
                "downstream_pipeline_wall_ms": row["timing"]["end_to_end_ms"],
                "generation_cache_hit": row["generation_cache_hit"], "original_generation_ms": row["original_generation_ms"],
                "judge_status": "judged_against_original_gold" if score else "not_run",
                **{key: score.get(key) for key in ("correctness", "grounding", "completeness", "normalised_answer_score")},
            })
    write_csv(output/"rag_comparison.csv", combined)
    write_json_atomic(output/"rag_comparison.json", combined)
    return combined
