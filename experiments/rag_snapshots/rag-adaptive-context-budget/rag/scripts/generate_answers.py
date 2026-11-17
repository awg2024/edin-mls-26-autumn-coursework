
"""Generate answer candidates using the audio-to-answer baseline."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
from src.asr_adapter import ASRAdapter

from src.context_builder import (SimpleContextBuilder, ExperimentalContextBuilder)

from src.reranker import CrossEncoderReranker

from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever
from extensions.adaptive_context_budget.integration import add_budget_arguments, wrap_from_arguments

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Diagnostic subset only")
    
    ap.add_argument("--context-strategy",choices=["simple", "tfidf", "mmr"],default="simple",help="Context construction algorithm")
    ap.add_argument("--max-context-chars",type=int,default=5000,help="Maximum evidence characters for experimental strategies")
    ap.add_argument("--mmr-lambda",type=float,default=0.7,help="MMR relevance/diversity trade-off")

    ap.add_argument("--reranker-model", type=Path, default=None, help="Path to a locally available cross-encoder model")
    ap.add_argument("--rerank-k", type=int, default=5, help="Number of documents kept after reranking")
    add_budget_arguments(ap)

    args = ap.parse_args()

    if args.max_context_chars < 1:
        ap.error("--max-context-chars must be positive")

    if not 0 <= args.mmr_lambda <= 1:
        ap.error("--mmr-lambda must be between 0 and 1")

    if args.rerank_k < 1 or args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("rank-r, --k and --limit must be positive")

    if args.reranker_model is not None:
        if args.rerank_k > args.k:
            ap.error("--rerank-k cannot exceed --k")

        if not args.reranker_model.is_dir():
            ap.error("--reranker-model must be an existing directory")

    if args.output.exists():
        ap.error("Output exists; choose a new filename to preserve prior results")

    asr = ASRAdapter()
    asr.prepare()
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)

    if args.context_strategy == "simple":
        context_builder = SimpleContextBuilder()
    else:
        context_builder = ExperimentalContextBuilder(
            strategy=args.context_strategy,
            max_chars=args.max_context_chars,
            mmr_lambda=args.mmr_lambda)


    try:
        context_builder = wrap_from_arguments(context_builder, args)
    except ValueError as exc:
        ap.error(str(exc))

    reranker = None
    if args.reranker_model is not None:
        reranker = CrossEncoderReranker(model_path=args.reranker_model, device=args.device or "cpu", batch_size=8)

    pipeline = RAGPipeline(asr, LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, context_builder, ELMGenerator(capture_usage=args.context_budget != "none"), reranker=reranker)
    


    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for item in items:
            result = pipeline.answer_audio(data_dir / item["audio_file"], k=args.k, final_k=args.rerank_k)
            row = {"query_id": item["query_id"], 
                   "split": args.split, 
                   "audio_file": item["audio_file"],
                   "transcript": result["transcript"],
                     "answer": result["answer"],
                   "retrieved_record_ids": result["retrieval"]["record_ids"], 
                   "timing": result["timing"],
                   "retrieval_scores": result["retrieval"]["scores"],
                    "context_strategy": args.context_strategy,
                    "k": args.k,
                    "contexts": result["contexts"],
                    "prompt": result["prompt"],
                    "selected_evidence": result["selected_evidence"],
                    "candidate_k": args.k,
                    "reranker_enabled": reranker is not None,
                    "final_retrieved_record_ids": result["final_retrieval"]["record_ids"],
                    "final_retrieval_scores": result["final_retrieval"]["scores"],
                    "rerank_ms": result["timing"]["rerank_ms"]}
 
            if "budget_metrics" in result:
                row["budget_metrics"] = result["budget_metrics"]
                row["generation_usage"] = result.get("generation_usage")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(item["query_id"])

if __name__ == "__main__":
    main()
