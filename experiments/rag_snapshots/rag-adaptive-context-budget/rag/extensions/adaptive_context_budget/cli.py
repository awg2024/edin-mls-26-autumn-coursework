"""Explicit offline/real-generation commands; API calls are never the default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .core import AdaptiveBudget, TokenCounter
from .analysis import plot_results, summarise
from .benchmark import benchmark
from .experiment import (
    make_builders, make_conditions, prepare_query, read_jsonl, run_comparison,
    stable_hash, validate_snapshot, write_csv, write_json,
)

RAG_ROOT = Path(__file__).resolve().parents[2]


def prepare(args):
    if args.output.exists():
        raise ValueError("snapshot exists; choose a new filename")
    if args.k < 1 or args.limit < 1 or args.max_chars < 1:
        raise ValueError("k, limit and max-chars must be positive")
    required = [args.data_dir / "corpus.jsonl", args.data_dir / "student_audio_manifest.jsonl",
                args.data_dir / "embeddings/corpus_enc_a.npy",
                args.data_dir / "embeddings/corpus_enc_a_ids.json", args.transcript_cache]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError("Missing course inputs (no synthetic replacement): " + ", ".join(missing))
    from src.data import CourseData
    from scripts.evaluate_rag import load_cache
    from src.encoder import LocalEncoder
    from src.retriever import ExactDenseRetriever

    data = CourseData(args.data_dir)
    items = data.audio_items(args.split)[:args.limit]
    cache = load_cache(args.transcript_cache)
    missing_queries = [item["query_id"] for item in items if item["query_id"] not in cache]
    if missing_queries:
        raise ValueError("Cache is incomplete; first missing IDs: " + ", ".join(missing_queries[:10]))
    if not items:
        raise ValueError("no queries in selected split")
    builders = make_builders(args.selectors, args.max_chars, args.mmr_lambda)
    encoder = LocalEncoder(args.encoder, device=args.device)
    retriever = ExactDenseRetriever(args.data_dir / "embeddings/corpus_enc_a.npy",
                                    args.data_dir / "embeddings/corpus_enc_a_ids.json")
    reranker = None
    if args.reranker_model:
        if not 1 <= args.final_k <= args.k:
            raise ValueError("reranker final-k must be between 1 and k")
        from src.reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(args.reranker_model, device=args.device)
    rows = []
    for item in items:
        rows.append(prepare_query(item, cache[item["query_id"]], data, encoder, retriever,
                                  builders, args.k, reranker, args.final_k))
        print(f"Prepared {item['query_id']}", flush=True)
    validate_snapshot(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RAG_ROOT,
                              capture_output=True, text=True, check=False)
    write_json(args.output.with_suffix(".meta.json"), {
        "split": args.split, "queries": len(rows), "k": args.k,
        "final_k": args.final_k if reranker else args.k,
        "max_context_chars": args.max_chars, "mmr_lambda": args.mmr_lambda,
        "selectors": args.selectors, "encoder": str(args.encoder),
        "reranker_model": str(args.reranker_model) if args.reranker_model else None,
        "transcript_cache": str(args.transcript_cache), "transcript_cache_hash": stable_hash(cache),
        "snapshot_hash": stable_hash(rows), "git_head": revision.stdout.strip(),
        "asr_mode": "cached transcripts only; no repeated ASR", "qrels_used_for_preparation": False,
    })
    from .integration import extract_evidence, control_metrics
    lengths = []
    for row in rows:
        for selector, selection in row["selections"].items():
            metrics = control_metrics(extract_evidence(selection["built"], row["final_retrieval"]["record_ids"]))
            lengths.append({"query_id": row["query_id"], "selector": selector,
                            "estimated_evidence_tokens": metrics["evidence_tokens_before"],
                            "units": metrics["candidate_units"], "counter": metrics["token_counter"]})
    write_csv(args.output.with_suffix(".lengths.csv"), lengths)
    print(f"Saved {len(rows)} frozen queries and a context-length inspection CSV")


def run(args):
    queries = read_jsonl(args.snapshot)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        queries = queries[:args.limit]
    validate_snapshot(queries)
    if args.generate:
        if not args.conditions:
            raise ValueError("API generation requires an explicit --conditions list")
        if args.limit is None or (args.limit > 5 and not args.allow_large_api_run):
            raise ValueError("Start with --limit 5; larger API runs require --allow-large-api-run")
    counter = TokenCounter.from_local_tokenizer(args.tokenizer_path) if args.tokenizer_path else TokenCounter()
    adaptive = AdaptiveBudget(args.min_budget, args.max_budget, args.low_confidence, args.high_confidence, args.gamma)
    specs = make_conditions(queries, args.budgets, args.packing_policies, adaptive, counter, args.match_adaptive)
    available = {spec.name: spec for spec in specs}
    if args.conditions:
        unknown = set(args.conditions) - set(available)
        if unknown:
            raise ValueError("Unknown conditions: " + ", ".join(sorted(unknown)))
        specs = [available[name] for name in args.conditions]
    print("Conditions:", ", ".join(spec.name for spec in specs))
    print(f"Queries: {len(queries)}; planned generation calls: {len(queries) * len(specs) if args.generate else 0}")
    if args.output.exists():
        raise ValueError("output directory exists; choose a new directory")
    qrels = None
    if args.qrels:
        from src.evaluation.answer_judge import load_qrels
        qrels = load_qrels(args.qrels)  # Used only below for post-selection evaluation.
    generator = None
    if args.generate:
        from dotenv import load_dotenv
        from src.generator import ELMGenerator
        load_dotenv(RAG_ROOT / ".env")
        generator = ELMGenerator(capture_usage=True)
    meta_path = args.snapshot.with_suffix(".meta.json")
    provenance = json.loads(meta_path.read_text()) if meta_path.exists() else {"snapshot_path": str(args.snapshot)}
    run_comparison(queries, specs, args.output, counter, generator, qrels, provenance)
    summarise(args.output)
    print(f"Results saved in {args.output}; answer quality requires the supplied evaluator")


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze cached transcripts, retrieval and existing selectors once")
    prep.add_argument("--data-dir", type=Path, default=RAG_ROOT / "data/rag_corpus")
    prep.add_argument("--encoder", type=Path, default=RAG_ROOT / "models/enc_a")
    prep.add_argument("--transcript-cache", type=Path, required=True)
    prep.add_argument("--split", choices=["practice", "dev"], default="practice")
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--limit", type=int, default=5)
    prep.add_argument("--k", type=int, default=5)
    prep.add_argument("--selectors", nargs="+", choices=["simple", "tfidf", "mmr"], default=["simple", "tfidf", "mmr"])
    prep.add_argument("--max-chars", type=int, default=5000)
    prep.add_argument("--mmr-lambda", type=float, default=0.7)
    prep.add_argument("--reranker-model", type=Path)
    prep.add_argument("--final-k", type=int, default=5)
    prep.add_argument("--device", default="cpu")
    prep.set_defaults(action=prepare)

    experiment = sub.add_parser("run", help="Offline packing by default; real generation requires --generate")
    experiment.add_argument("--snapshot", type=Path, required=True)
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--limit", type=int)
    experiment.add_argument("--budgets", nargs="+", type=int, default=[256, 512, 1024, 2048])
    experiment.add_argument("--packing-policies", nargs="+", choices=["skip", "sentence_prefix"], default=["skip", "sentence_prefix"])
    experiment.add_argument("--min-budget", type=int, default=256)
    experiment.add_argument("--max-budget", type=int, default=2048)
    experiment.add_argument("--low-confidence", type=float, default=0.1)
    experiment.add_argument("--high-confidence", type=float, default=0.6)
    experiment.add_argument("--gamma", type=float, default=1.0)
    experiment.add_argument("--match-adaptive", action="store_true")
    experiment.add_argument("--tokenizer-path")
    experiment.add_argument("--conditions", nargs="+")
    experiment.add_argument("--qrels", type=Path, help="Optional evaluation-only qrels; never supplied to budget policy")
    experiment.add_argument("--generate", action="store_true")
    experiment.add_argument("--allow-large-api-run", action="store_true")
    experiment.set_defaults(action=run)

    summary = sub.add_parser("summarise", help="Join existing judge results and compute paired comparisons")
    summary.add_argument("--run-dir", type=Path, required=True)
    summary.set_defaults(action=lambda args: summarise(args.run_dir))
    overhead = sub.add_parser("overhead", help="Explicit synthetic CPU packing microbenchmark; no API calls")
    overhead.add_argument("--output", type=Path, required=True)
    overhead.add_argument("--counts", nargs="+", type=int, default=[1, 5, 20, 100, 500, 1000])
    overhead.add_argument("--repeats", type=int, default=30)
    overhead.add_argument("--warmups", type=int, default=5)
    overhead.set_defaults(action=lambda args: benchmark(args.output, args.counts, args.repeats, args.warmups))
    plots = sub.add_parser("plot", help="PNG/PDF figures from actual measurements; missing axes are skipped")
    plots.add_argument("--output", type=Path, required=True)
    plots.add_argument("--run-dir", type=Path)
    plots.add_argument("--overhead-csv", type=Path)
    plots.set_defaults(action=lambda args: plot_results(args.output, args.run_dir, args.overhead_csv))
    return ap


def main(argv=None):
    ap = parser()
    args = ap.parse_args(argv)
    try:
        args.action(args)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError) as exc:
        ap.error(str(exc))


if __name__ == "__main__":
    main()
