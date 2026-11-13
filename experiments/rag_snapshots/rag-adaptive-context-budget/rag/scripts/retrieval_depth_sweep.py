# scripts/retrieval_depth_sweep.py

from time import perf_counter
from src.metrics import recall_at_k, hit_at_k

K_VALUES = [1, 5, 10, 20, 30, 40, 50] # k sweep for cosine similarity  


def evaluate_query(query_id, transcript,encoder,retriever,qrels):
    
    # Ground-truth relevant document IDs from eval  
    relevant_ids = set(qrels.get(query_id, {}))

    # Encode the transcript once
    t0 = perf_counter()
    query_vector = encoder.encode(transcript)
    embedding_ms = (perf_counter() - t0) * 1000

    # Retrieve 50 (max) candidate documents once 
    result = retriever.retrieve(query_vector, k=max(K_VALUES))

    ranked_ids = result["record_ids"]
    rows = []

    for k in K_VALUES:
        
        # Only use the first K ranked documents
        top_k_ids = ranked_ids[:k]

        rows.append({
            "query_id": query_id,
            "k": k,
            "recall": (recall_at_k(ranked_ids, relevant_ids, k) if relevant_ids else None),  # recall@K  Of all the documents known to be relevant, what proportion did we retrieve within the first K results?
            "hit": hit_at_k(ranked_ids, relevant_ids, k),  # hit@K  Did we retrieve at least one relevant document within the first K results?
            "retrieved_ids": top_k_ids,
            "embedding_ms": embedding_ms,
        })

    return rows, result
