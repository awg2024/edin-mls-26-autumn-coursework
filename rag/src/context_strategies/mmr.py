
# src/context_strategies/mmr.py 

import numpy as np
from .tfidf import prepare_tfidf
from .utils import fits_budget


def select_mmr(query, documents, max_chars=5000, lambda_=0.7):
    """
    Greedy Maximal Marginal Relevance selection.

    lambda_ = 1.0 -> relevance only
    lambda_ = 0.0 -> strongest diversity preference
    """

    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between 0 and 1")
    
    candidates, vectors, relevance = prepare_tfidf(query, documents) # run tfidf to get an embedding and similarity score 

    if vectors is None:  
        return []
    
    selected = []

    remaining = {  # Only consider candidates with positive relevance given by tfidf 
        i for i in range(len(candidates))
        if relevance[i] > 0
    }

    # For each candidate, remember its highest
    # similarity to any already-selected sentence.
    max_redundancy = np.zeros(len(candidates))

    while remaining:  # greedy selection for each group, as long as there are candidate sentneces lfet 
        best_index = None
        best_score = float("-inf") # initialise negative infinity

        for i in sorted(remaining): # sort in meaningful order 
            candidate = candidates[i]

            if not fits_budget(candidate, selected, max_chars): #  check the document fits budget (5k) 
                continue

            if not selected: # select the first sentence. we cannot comapre redudancy for this first one since there's nothing to compare against 
                score = float(relevance[i])
            else:
                score = (lambda_ * relevance[i] - (1 - lambda_) * max_redundancy[i]) #  score for MMR 

            if score > best_score: 
                best_score = float(score)
                best_index = i

        if best_index is None:
            break  # No remaining evidence fits.

        # Copy so we don't modify the shared candidates.
        chosen = dict(candidates[best_index])

        chosen["relevance_score"] = float(relevance[best_index])
        chosen["mmr_score"] = best_score

        selected.append(chosen)
        remaining.remove(best_index)

        # Compare every candidate with the newly selected sentence.
        similarities = (vectors @ vectors[best_index].T).toarray().ravel() #  sparse flatten vector for each similarity 

        # compares the two arrays element by element and keeps whichever value is larger for each sentnece. 
        max_redundancy = np.maximum(max_redundancy, similarities)

    return selected
