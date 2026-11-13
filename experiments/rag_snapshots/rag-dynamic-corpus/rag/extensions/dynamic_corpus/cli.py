"""Explicit CPU benchmarks and opt-in small ELM trial. No model calls by default."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .benchmark import Case, run_suite, suite_cases, validate_checkpoint
from .index import DynamicCorpus
from .io import fingerprint, load_jsonl, write_json_atomic
from .workload import Catalog, Queries, apply_oracle, mutation_from_entry

ROOT = Path(__file__).resolve().parents[2]


def _catalog(args):
    if args.dataset == "synthetic":
        return Catalog.synthetic(args.synthetic_records, args.dimension, args.seed)
    return Catalog.course(args.data_dir)


def prepare_queries(args):
    from scripts.evaluate_rag import load_cache
    from src.data import CourseData
    from src.encoder import LocalEncoder

    if args.output.exists() or args.output.with_suffix(".meta.json").exists():
        raise ValueError("query output exists; use a new output name")
    data = CourseData(args.data_dir)
    items = data.audio_items(args.split)[:args.limit]
    if not items:
        raise ValueError("no matching audio queries")
    cache = load_cache(args.transcript_cache)
    missing = [item["query_id"] for item in items if item["query_id"] not in cache]
    if missing and not args.transcribe_missing:
        raise ValueError("missing cached real ASR transcripts; supply --transcribe-missing explicitly: " + str(missing))
    if missing:
        from src.asr_adapter import ASRAdapter
        from scripts.evaluate_rag import append_cache
        asr = ASRAdapter()
        for item in items:
            qid = item["query_id"]
            if qid not in cache:
                cache[qid] = asr.transcribe(args.data_dir/item["audio_file"])
                append_cache(args.transcript_cache, qid, cache[qid])
    transcripts = [cache[item["query_id"]] for item in items]
    if any(not isinstance(text, str) or not text.strip() for text in transcripts):
        raise ValueError("empty or invalid ASR transcript")
    encoder = LocalEncoder(args.model_dir, device=args.device)
    vectors = encoder.encode_many(transcripts)  # Queries only; never used as a document encoder.
    queries = Queries([item["query_id"] for item in items], vectors, "real_asr_local_enc_a_query_cache",
                      "enc_a", transcripts, [item["audio_file"] for item in items], args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vectors=queries.vectors, query_ids=np.asarray(queries.ids))
    write_json_atomic(args.output.with_suffix(".meta.json"), {
        "source": queries.source, "embedding_space": queries.embedding_space, "split": queries.split,
        "transcripts": queries.transcripts, "audio_files": queries.audio_files,
        "encoder_model": str(args.model_dir), "transcript_cache": str(args.transcript_cache),
    })
    print(args.output)


def replay(args):
    config = json.loads((args.case_dir/"config.json").read_text())
    catalog = _catalog(args).subset(config["corpus_size"], config["seed"])
    if catalog.fingerprint() != config["catalog_fingerprint"]:
        raise ValueError("replay catalog fingerprint mismatch (wrong source/seed/dimension)")
    ids = config["initial_ids"]
    index = DynamicCorpus({rid: catalog.records[rid] for rid in ids},
                          catalog.vectors[[catalog.positions[rid] for rid in ids]], ids,
                          embedding_space=catalog.embedding_space)
    log = load_jsonl(args.case_dir/"mutations.jsonl")
    if fingerprint(log) != config["mutation_log_hash"]:
        raise ValueError("mutation log fingerprint mismatch")
    oracle = {rid: (catalog.records[rid], catalog.vector(rid)) for rid in ids}
    sources = {rid: rid for rid in ids}
    for start in range(0, len(log), config["batch_size"]):
        entries = log[start:start+config["batch_size"]]
        index.apply_batch([mutation_from_entry(catalog, entry) for entry in entries])
        for entry in entries:
            apply_oracle(oracle, sources, catalog, entry)
    with np.load(args.case_dir/"queries.npz", allow_pickle=False) as arrays:
        queries = Queries(arrays["query_ids"].tolist(), arrays["vectors"].copy(),
                          config["query_source"], config["embedding_space"])
    checks = validate_checkpoint(index, oracle, sources, catalog, queries, config["k"])
    if not all(r["exact_topk_agreement"] and r["scores_within_tolerance"] and
               r["logical_state_equal"] and r["all_active_and_current_text"] for r in checks):
        raise AssertionError("replay/reference mismatch")
    print(json.dumps({"logical_digest": index.snapshot().digest(), "stats": index.get_stats(),
                      "query_checks": checks}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("benchmark", "ablate", "replay", "rag-trial"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", choices=("course", "synthetic"), default="course")
        p.add_argument("--data-dir", type=Path, default=ROOT/"data/rag_corpus")
        p.add_argument("--synthetic-records", type=int, default=24000)
        p.add_argument("--dimension", type=int, default=64)
        p.add_argument("--seed", type=int, default=2026)
        if command == "replay":
            p.add_argument("--case-dir", type=Path, required=True)
            continue
        p.add_argument("--queries", type=Path, help="Real enc_a query cache produced by prepare-queries")
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--k", type=int, default=10)
        if command == "rag-trial":
            p.add_argument("--generate", action="store_true", help="Explicitly authorise a small real ELM generation trial")
            p.add_argument("--limit", type=int, default=5)
            p.add_argument("--allow-large-api-run", action="store_true")
            p.add_argument("--operations", type=int, default=20)
            p.add_argument("--final-k", type=int, default=5)
            p.add_argument("--selector", choices=("simple", "tfidf", "mmr"), default="simple")
            p.add_argument("--rerank", action="store_true")
            p.add_argument("--reranker-model", type=Path)
            continue
        p.add_argument("--probes", type=int, default=4)
        p.add_argument("--repeats", type=int, default=3)
        p.add_argument("--no-memory-trace", action="store_true")
        p.add_argument("--plot", action="store_true")
        if command == "ablate":
            p.add_argument("--base-size", type=int, default=2000)
            p.add_argument("--corpus-sizes", type=int, nargs="+", default=[2000, 5000, 10000, 24000])
        else:
            for name, default in (("corpus-size", 2000), ("operations", 200), ("batch-size", 1),
                                  ("queries-per-update", 1), ("checkpoint-every", 50)):
                p.add_argument("--" + name, type=int, default=default)
            p.add_argument("--compaction-threshold", type=float, default=0.05)
            p.add_argument("--initial-fraction", type=float, default=0.8)
            p.add_argument("--profile", choices=("insert", "update", "delete", "mixed"), default="mixed")
            p.add_argument("--update-mode", choices=("refresh", "replace"), default="refresh")
    p = sub.add_parser("prepare-queries")
    p.add_argument("--data-dir", type=Path, default=ROOT/"data/rag_corpus")
    p.add_argument("--model-dir", type=Path, default=ROOT/"models/enc_a")
    p.add_argument("--transcript-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("practice", "dev"), default="practice")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--device", default="cpu")
    p.add_argument("--transcribe-missing", action="store_true")
    p = sub.add_parser("plot")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("summarise-trial")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-queries":
            if args.limit < 1 or args.output.suffix != ".npz":
                raise ValueError("positive limit and .npz output required")
            prepare_queries(args)
            return
        if args.command == "plot":
            from .plots import plot
            plot(args.output)
            return
        if args.command == "summarise-trial":
            from .trial import collect_trial_scores
            collect_trial_scores(args.output)
            return
        if args.command == "replay":
            replay(args)
            return
        if args.output.exists():
            raise ValueError("experiment output exists; use a new path to preserve raw evidence")
        catalog = _catalog(args)
        if args.command == "rag-trial":
            if not args.generate or not args.queries:
                raise ValueError("rag-trial requires --generate and --queries from real ASR/enc_a outputs")
            if args.limit < 1 or args.limit > 5 and not args.allow_large_api_run:
                raise ValueError("start with 1–5 queries; larger runs need --allow-large-api-run")
            queries = Queries.load(args.queries)
            queries = Queries(queries.ids[:args.limit], queries.vectors[:args.limit], queries.source, queries.embedding_space,
                              queries.transcripts[:args.limit] if queries.transcripts else None,
                              queries.audio_files[:args.limit] if queries.audio_files else None, queries.split)
            from src.context_builder import SimpleContextBuilder, ExperimentalContextBuilder
            from src.generator import ELMGenerator
            from .trial import run_trial
            builder = SimpleContextBuilder() if args.selector == "simple" else ExperimentalContextBuilder(strategy=args.selector)
            reranker = None
            if args.rerank:
                if not args.reranker_model:
                    raise ValueError("--rerank requires the existing local --reranker-model directory")
                from src.reranker import CrossEncoderReranker
                reranker = CrossEncoderReranker(args.reranker_model)
            run_trial(catalog, queries, args.output, ELMGenerator(), builder, operations=args.operations,
                      seed=args.seed, k=args.k, final_k=args.final_k, reranker=reranker)
            return
        queries = Queries.load(args.queries) if args.queries else Queries.probes(catalog, args.probes, args.seed)
        qrels = None
        if args.dataset == "course" and args.queries and queries.split:
            from src.data import CourseData
            qrels = CourseData(args.data_dir).load_qrels(queries.split)
        if args.command == "ablate":
            cases = suite_cases(args.base_size, args.corpus_sizes, args.seed)
        else:
            cases = [Case(**{key: getattr(args, key) for key in asdict(Case()) if key != "family"})]
        run_suite(catalog, queries, cases, args.output, args.repeats, args.k, qrels, not args.no_memory_trace)
        if args.plot:
            from .plots import plot
            plot(args.output)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as error:
        parser.exit(2, f"Not executed: {error}\n")


if __name__ == "__main__":
    main()
