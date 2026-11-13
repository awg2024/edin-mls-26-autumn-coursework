from __future__ import annotations

import time
from pathlib import Path
import json
import numpy as np


def normalise_query(query_vector, dimension):
    """Shared query validation/normalisation for static and dynamic exact search."""
    q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if q.size != dimension or not np.isfinite(q).all():
        raise ValueError("query vector must be finite and match embedding dimension")
    norm = np.linalg.norm(q)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("zero or non-finite query vector norm")
    return q / norm


def exact_scores(embeddings, normalised_query, reproducible=False):
    """Exact FP32 dot products; opt-in row reduction is partition-independent.

    The original static default remains BLAS matrix-vector multiplication.
    Dynamic/rebuilt controls opt into the same row-wise summation so changing
    the base/delta partition cannot change score ties through BLAS roundoff.
    """
    if reproducible:
        return np.einsum("ij,j->i", embeddings, normalised_query, dtype=np.float32, optimize=False)
    return embeddings @ normalised_query


def top_k_indices(scores, record_ids, k, deterministic_ties=False):
    """Legacy ranking by default; opt-in exact-score ties use ascending ID."""
    k = min(k, len(scores))
    if not k:
        return np.empty(0, dtype=np.intp)
    if not deterministic_ties:
        indices = np.argpartition(-scores, k - 1)[:k]
        return indices[np.argsort(-scores[indices])]
    ids = np.asarray(record_ids, dtype=object)
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    higher = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)
    tied = tied[np.argsort(ids[tied], kind="stable")[:k - len(higher)]]
    chosen = np.concatenate((higher, tied))
    return chosen[np.lexsort((ids[chosen], -scores[chosen]))]


class ExactDenseRetriever:
    """Transparent exact cosine-similarity baseline over precomputed corpus vectors."""

    def __init__(self, embeddings_path: str | Path, ids_path: str | Path):
        self.embeddings = np.load(embeddings_path) # load precomputed document embeddings 
        self.record_ids = json.loads(Path(ids_path).read_text(encoding="utf-8")) # load corresponding document IDs (for origin corpus lookup)

        if len(self.record_ids) != self.embeddings.shape[0]:
            raise ValueError("embedding row count does not match ID list")

        self.embeddings = np.asarray(self.embeddings, dtype=np.float32) # ensure embeddings use a float32 value

        norms = np.linalg.norm(self.embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4): # embeddings should have been regularized, with len=1
            raise ValueError("corpus embeddings are expected to be L2-normalised")

    @classmethod
    def from_arrays(cls, embeddings, record_ids, *, deterministic_ties=False, reproducible=False):
        """In-memory index construction without writing experimental assets."""
        matrix = np.array(embeddings, dtype=np.float32, order="C", copy=True)
        ids = list(record_ids)
        if matrix.ndim != 2 or matrix.shape[1] < 1 or len(ids) != matrix.shape[0]:
            raise ValueError("embedding matrix shape must match IDs and have positive dimension")
        if any(not isinstance(rid, str) or not rid.strip() for rid in ids) or len(set(ids)) != len(ids):
            raise ValueError("record IDs must be nonempty, unique strings")
        if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4):
            raise ValueError("corpus embeddings must be finite and L2-normalised")
        instance = cls.__new__(cls)
        instance.embeddings = matrix
        instance.record_ids = ids
        instance.deterministic_ties = deterministic_ties
        instance.reproducible = reproducible
        return instance

    def retrieve(self, query_vector: np.ndarray, k: int = 5): # default depth retreival is 5. 
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer") 
        q = normalise_query(query_vector, self.embeddings.shape[1])

        k = min(k, len(self.record_ids))

        start = time.perf_counter()
        scores = exact_scores(self.embeddings, q, getattr(self, "reproducible", False))
        idx = top_k_indices(scores, self.record_ids, k, getattr(self, "deterministic_ties", False))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {
            "record_ids": [self.record_ids[int(i)] for i in idx], # which documents were retrieved?
            "scores": [float(scores[int(i)]) for i in idx], # their correspodning similarity scores
            "latency_ms": elapsed_ms, # how long retrieval took 
        }
