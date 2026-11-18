# scripts/generate_answers.py 

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

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Diagnostic subset only")
    
    # add in argparse options for experimental context builder 
    ap.add_argument("--context-strategy",choices=["simple", "tfidf", "mmr"],default="simple",help="Context construction algorithm")
    ap.add_argument("--max-context-chars",type=int,default=5000,help="Maximum evidence characters for experimental strategies")
    ap.add_argument("--mmr-lambda",type=float,default=0.7,help="MMR relevance/diversity trade-off")  # defining lmbda for mmr 

    # add in argparse options for reranking 
    ap.add_argument("--reranker-model", type=Path, default=None, help="Path to a locally available cross-encoder model")
    ap.add_argument("--rerank-k", type=int, default=5, help="Number of documents kept after reranking")

    args = ap.parse_args()

    # validation of experimental context builder after parsing 
    if args.max_context_chars < 1:  # validation for experimental context builder 
        ap.error("--max-context-chars must be positive")

    if not 0 <= args.mmr_lambda <= 1:
        ap.error("--mmr-lambda must be between 0 and 1")

    # validation of the reranking option after parsing  
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

    # before creating the RAG pipeline check the context builder passing 
    if args.context_strategy == "simple": # enable context builder to handle simple (baseline) 
        context_builder = SimpleContextBuilder()
    else:
        context_builder = ExperimentalContextBuilder( # enable context builder to handle experimental context builder 
            strategy=args.context_strategy,
            max_chars=args.max_context_chars,
            mmr_lambda=args.mmr_lambda)


    # before building the context builder check reranking isn't passed
    reranker = None # init 
    if args.reranker_model is not None:
        reranker = CrossEncoderReranker(model_path=args.reranker_model,  # expect models/cross_encoder downloaded from hugging face 
                                        device=args.device or "cpu", batch_size=8)

    # RAG pipeline call 
    pipeline = RAGPipeline(asr, LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, context_builder, ELMGenerator(), reranker=reranker)  # context_builder imbedded in RAG pipeline call 
    


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
                   "retrieval_scores": result["retrieval"]["scores"],  #  update output row for experimental deocder 
                    "context_strategy": args.context_strategy,
                    "k": args.k,
                    "contexts": result["contexts"],
                    "prompt": result["prompt"],
                    "selected_evidence": result["selected_evidence"],
                    "candidate_k": args.k,
                    "reranker_enabled": reranker is not None,  # ranking output for testing. 
                    "final_retrieved_record_ids": result["final_retrieval"]["record_ids"],  # state of these if baseline is enabled? 
                    "final_retrieval_scores": result["final_retrieval"]["scores"],
                    "rerank_ms": result["timing"]["rerank_ms"],
                    "final_k": len(result["final_retrieval"]["record_ids"]),
                    "rerank_k": args.rerank_k if reranker is not None else None,
                    "reranker_model": str(args.reranker_model) if reranker is not None else None,
                    "reranker_device": args.reranker_device if reranker is not None else None,
                    "max_context_chars": (
                        args.max_context_chars if args.context_strategy != "simple" else None
                    ),
                    "mmr_lambda": args.mmr_lambda if args.context_strategy == "mmr" else None,
                    "context_chars": sum(len(c) for c in result["contexts"])}
 
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(item["query_id"])

if __name__ == "__main__":
    main()
