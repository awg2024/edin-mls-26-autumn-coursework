# src/metrics.py

from __future__ import annotations
import math


def recall_at_k(ranked_ids, relevant_ids, k): # measure the proportion of relevant documents in top k 
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / len(relevant) 

def mrr_at_k(ranked_ids, relevant_ids, k=10): # evaluates the first relevant document appears in the ranked list (3, perfect / 1, slightly relevant). 
    relevant = set(relevant_ids)
    for rank, rid in enumerate(ranked_ids[:k], start=1):
        if rid in relevant:
            return 1.0 / rank # reciprocal rank (1st, 2nd) weighting 
    return 0.0



def hit_at_k(ranked_ids, relevant_ids, k): # amount of relevant tokens given the number of tokens we loaded in. 
    relevant = set(relevant_ids)

    return float(any(rid in relevant for rid in ranked_ids[:k]))



def ndcg_at_k(ranked_ids, relevance, k=10): # graded relevance 
    def gain(rel, rank):
        return (2.0 ** rel - 1.0) / math.log2(rank + 1.0) # calculate exponential cumulative gain

    dcg = sum( # sorts all available non-zero relevance scores in descending order
        gain(float(relevance.get(rid, 0.0)), rank)
        for rank, rid in enumerate(ranked_ids[:k], start=1)
        if float(relevance.get(rid, 0.0)) > 0
    )

    ideal_rels = sorted( # normalise scores between 0-1
        [float(v) for v in relevance.values() if float(v) > 0],
        reverse=True,
    )[:k]
    idcg = sum(gain(rel, rank) for rank, rel in enumerate(ideal_rels, start=1))

    return dcg / idcg if idcg > 0 else 0.0


# utilising top-k sampling then sentnece level re-ranking compression for later stages to get exact sentences? 