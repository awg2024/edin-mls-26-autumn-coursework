# scripts/evaluate_rag.py  

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
import numpy as np

from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from src.metrics import (recall_at_k, hit_at_k, mrr_at_k, ndcg_at_k)


def load_cache(path: Path):
    cache = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    cache[row["query_id"]] = row["transcript"]
    return cache


def append_cache(path: Path, qid: str, transcript: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"query_id": qid, "transcript": transcript},
            ensure_ascii=False,
        ) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--transcript-cache", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k-max", type=int, default=100)
    
    ap.add_argument("--sweep-k",type=int,nargs="+",default=[1, 5, 10, 20, 30, 40, 50]) # sweep-k retrieval experiment, nargs+ shows we can have multiple k arguments 
    ap.add_argument("--output", type=Path, default=None, help="Save per-query metrics to JSONL") # json file output for report writupe
    args = ap.parse_args()


    required_k = max(max(args.sweep_k),10)

    if args.k_max < required_k:
        ap.error(f"--k-max must be at least {required_k}") # for specified k recal..  

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    qrels = data.load_qrels(args.split)

    asr = ASRAdapter()
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings" / "corpus_enc_a.npy",
        data_dir / "embeddings" / "corpus_enc_a_ids.json")

    cache = load_cache(args.transcript_cache) if args.transcript_cache else {}

    rows = []
    for i, item in enumerate(items, start=1):
        qid = item["query_id"]
        audio = data_dir / item["audio_file"]

        if qid in cache:
            transcript = cache[qid]
        else:
            transcript = asr.transcribe(audio)
            if args.transcript_cache:
                append_cache(args.transcript_cache, qid, transcript)

        qvec = encoder.encode(transcript)
        ret = retriever.retrieve(qvec, k=args.k_max)
        ranked = ret["record_ids"]
        rel = qrels[qid]

        # output structure 
        row = {"query_id": qid}
        for k in args.sweep_k:
            row[f"Recall@{k}"] = recall_at_k(ranked, rel, k) # adaptive recall and hit depending on k being passed 
            row[f"Hit@{k}"] = hit_at_k(ranked, rel, k)
        row["MRR@10"] = mrr_at_k(ranked, rel, 10)
        row["NDCG@10"] = ndcg_at_k(ranked, rel, 10)
        rows.append(row)
        
        print(f"[{i:>3}/{len(items)}] {qid} R@10={row['Recall@10']:.3f}")


    keys = [] # keys dict define 

    for k in args.sweep_k:
        keys.extend([f"Recall@{k}", f"Hit@{k}"]) #  make hit & recall adaptive based on K value, not hard coding. 

    keys.extend(["MRR@10", "NDCG@10"]) # extend recall & hit to include context strategies at k=10 
    
    print("\nAggregate:")
    for key in keys: #  for every keys inside of top k? 

        values = [r[key] for r in rows]
        print(f"{key:<12} {np.mean(values):.6f}")


    if args.output is not None: 
        
        if args.output.exists():
            ap.error("Output already exists")

        args.output.parent.mkdir(parents=True, exist_ok=True) # save output dir 
        
        with args.output.open("x", encoding="utf-8") as f: 
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n") # json dump for k-sweep. 

if __name__ == "__main__":
    main()
