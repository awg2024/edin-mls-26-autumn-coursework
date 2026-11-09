# sr/context_strategies/tfidf.py  

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # import tf-idf vectorisor from sklearn 
from .utils import make_candidates, pack_ranked


def prepare_tfidf(query, documents):
    """
    Create candidate sentences and their TF-IDF
    relevance scores.

    Returns;;;
        candidates: sentence dictionaries
        vectors: sentence TF-IDF matrix
        scores: query-to-sentence cosine similarities
    """

    # call util function make canddiates 
    candidates = make_candidates(documents) # Return one candidate for each unique sentence.

    if not candidates:
        return candidates, None, np.array([])

    texts = [query] + [item["text"] for item in candidates] # extract out text from retrieve documents 

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), norm="l2") # word level tf-idf vectorisor using bigram & l2 norm

    try:
        matrix = vectorizer.fit_transform(texts)   # fit vectorisor to the text
    except ValueError as exc:
        if "empty vocabulary" not in str(exc):   # handle empty input cases 
            raise
        return candidates, None, np.zeros(len(candidates))

    # construct tf-idf matrix 
    query_vector = matrix[0:1] # Row 0 contains the query vector.
    sentence_vectors = matrix[1:] # All remaining rows contain sentence vectors.

    scores = (sentence_vectors @ query_vector.T).toarray().ravel() # Similarity score per setence sequence 
    return candidates, sentence_vectors, scores


def select_tfidf(query, documents, max_chars=5000):
    """Select query-relevant sentences under a budget."""

    candidates, _, scores = prepare_tfidf(query, documents)

    # Sort from highest similarity to lowest.
    ranked_indices = np.argsort(-scores, kind="stable")

    # Discard sentences with no lexical overlap (limitation talked about in the PR)
    ranked_indices = [i for i in ranked_indices if scores[i] > 0]

    selected = pack_ranked(candidates, ranked_indices, max_chars)

    # Attach relevance scores for experiment logging.
    for item in selected:
        index = candidates.index(item)
        item["relevance_score"] = float(scores[index])

    return selected
