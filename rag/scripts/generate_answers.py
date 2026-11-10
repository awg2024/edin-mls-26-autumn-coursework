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
    ap.add_argument("--mmr-lambda",type=float,default=0.7,help="MMR relevance/diversity trade-off")  # defining lmbda 

    args = ap.parse_args()

    if args.max_context_chars < 1:  # validation for experimental context builder 
        ap.error("--max-context-chars must be positive")

    if not 0 <= args.mmr_lambda <= 1:
        ap.error("--mmr-lambda must be between 0 and 1")


    if args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--k and --limit must be positive")
    if args.output.exists():
        ap.error("Output exists; choose a new filename to preserve prior results")
    asr = ASRAdapter()
    asr.prepare()
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    
    if args.context_strategy == "simple": # enable context builder to handle simple (baseline) 
        context_builder = SimpleContextBuilder()
    else:
        context_builder = ExperimentalContextBuilder( # enable context builder to handle experimental context builder 
            strategy=args.context_strategy,
            max_chars=args.max_context_chars,
            mmr_lambda=args.mmr_lambda)
        
    pipeline = RAGPipeline(asr, LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, context_builder, ELMGenerator())  # context_builder imbedded in RAG pipeline call 
    
    
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for item in items:
            result = pipeline.answer_audio(data_dir / item["audio_file"], k=args.k)
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
                    "selected_evidence": result["selected_evidence"]}
 
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(item["query_id"])

if __name__ == "__main__":
    main()
