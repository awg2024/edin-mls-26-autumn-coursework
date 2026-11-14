"""Compare dense top-5 with MiniLM top-5 from K=20 and K=50 candidates.

Uses previously cached audio transcripts: NO ASR, NO ELM API calls.
Outputs one JSONL result per query and flushes immediately for safe partial runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.reranker import CrossEncoderReranker
from src.retriever import ExactDenseRetriever


def metric_values(ids, qrels):
    return {
        "Recall@5": recall_at_k(ids, qrels, 5),
        "Hit@5": hit_at_k(ids, qrels, 5),
        "MRR@5": mrr_at_k(ids, qrels, 5),
        "NDCG@5": ndcg_at_k(ids, qrels, 5),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["practice", "dev"], default="practice")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoder-device", default="cuda")
    p.add_argument("--reranker-device", default="cuda")
    p.add_argument("--model", type=Path, default=ROOT / "models/cross_encoder")
    p.add_argument("--candidate-k", type=int, nargs="+", default=[20, 50])
    p.add_argument("--final-k", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if any(k < args.final_k for k in args.candidate_k) or args.final_k < 1:
        p.error("candidate K must be >= positive final K")
    if args.limit is not None and args.limit < 1:
        p.error("--limit must be positive")
    if args.output.exists():
        p.error("Output file exists; choose a different filename")

    cache = {}
    with args.cache.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cache[row["query_id"]] = row["transcript"]

    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit is not None:
        items = items[:args.limit]
    missing = [item["query_id"] for item in items if item["query_id"] not in cache]
    if missing:
        p.error(f"{len(missing)} query transcripts are missing from the cache; first missing: {missing[0]}")

    qrels = data.load_qrels(args.split)
    encoder = LocalEncoder(ROOT / "models/enc_a", device=args.encoder_device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings/corpus_enc_a.npy",
        data_dir / "embeddings/corpus_enc_a_ids.json",
    )
    reranker = CrossEncoderReranker(
        model_path=args.model, device=args.reranker_device, batch_size=8
    )

    summaries = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for i, item in enumerate(items, 1):
            qid = item["query_id"]
            transcript = cache[qid]
            t0 = time.perf_counter()
            query_vec = encoder.encode(transcript)
            embedding_ms = (time.perf_counter() - t0) * 1000
            dense = retriever.retrieve(query_vec, k=max(args.candidate_k))
            relevance = qrels.get(qid, {})
            baseline_ids = dense["record_ids"][:args.final_k]
            base_metrics = metric_values(baseline_ids, relevance)

            for k in args.candidate_k:
                candidate = {
                    "record_ids": dense["record_ids"][:k],
                    "scores": dense["scores"][:k],
                    "latency_ms": dense["latency_ms"],
                }
                t0 = time.perf_counter()
                ranked = reranker.rerank(
                    query=transcript, retrieval=candidate, data=data,
                    final_k=args.final_k,
                )
                rerank_ms = (time.perf_counter() - t0) * 1000
                rerank_metrics = metric_values(ranked["record_ids"], relevance)
                row = {
                    "query_id": qid,
                    "candidate_k": k,
                    "final_k": args.final_k,
                    "dense_top5_ids": baseline_ids,
                    "dense_metrics": base_metrics,
                    "reranked_ids": ranked["record_ids"],
                    "reranked_scores": ranked["scores"],
                    "reranked_metrics": rerank_metrics,
                    "embedding_ms": embedding_ms,
                    "dense_retrieval_ms": dense["latency_ms"],
                    "rerank_ms": rerank_ms,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                summaries.setdefault(k, []).append(row)
            print(f"[{i:>3}/{len(items)}] {qid} done", flush=True)

    print("\nAGGREGATE (per query averages):")
    for k, rows in sorted(summaries.items()):
        print(f"Candidate K={k}, rerank to {args.final_k}:")
        for label, key in (("Dense", "dense_metrics"), ("MiniLM", "reranked_metrics")):
            print(" ", label, " ".join(
                f"{m}={mean(r[key][m] for r in rows):.4f}"
                for m in ("Recall@5", "Hit@5", "MRR@5", "NDCG@5")
            ))
        print(f"  Mean rerank_ms={mean(r['rerank_ms'] for r in rows):.1f}")
    print(f"Saved {len(items) * len(args.candidate_k)} rows to {args.output}")


if __name__ == "__main__":
    main()
