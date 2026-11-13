"""Immutable base + append-only delta chunks, copy-on-write metadata and snapshots."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import sys
import threading
import time
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from src.retriever import ExactDenseRetriever, exact_scores, normalise_query, top_k_indices


def _freeze(array):
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Mutation:
    kind: str
    record_id: str
    text: str | None = None
    embedding: np.ndarray | None = None
    embedding_space: str | None = None


@dataclass(frozen=True)
class CompactionPolicy:
    delta_ratio: float | None = None
    tombstone_ratio: float | None = None

    def __post_init__(self):
        for value in (self.delta_ratio, self.tombstone_ratio):
            if value is not None and (not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError("compaction ratios must be in (0, 1], or None")


@dataclass(frozen=True)
class _State:
    version: int
    base: np.ndarray
    base_ids: np.ndarray
    base_positions: Mapping
    base_mask: np.ndarray
    base_active: int
    chunks: tuple[np.ndarray, ...]
    chunk_ids: tuple[np.ndarray, ...]
    chunk_masks: tuple[np.ndarray, ...]
    chunk_active: tuple[int, ...]
    delta_latest: Mapping
    texts: Mapping
    delta_rows: int = 0


def _embedding(state, rid):
    if rid not in state.texts:
        raise KeyError(rid)
    if rid in state.delta_latest:
        chunk, row = state.delta_latest[rid]
        return state.chunks[chunk][row]
    return state.base[state.base_positions[rid]]


def _materialise(state):
    ids = sorted(state.texts)
    dimension = state.base.shape[1]
    matrix = np.stack([_embedding(state, rid) for rid in ids]) if ids else np.empty((0, dimension), np.float32)
    return ids, matrix


def _search(state, query, k):
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    start = time.perf_counter()
    q = normalise_query(query, state.base.shape[1])
    candidates, candidate_scores = [], []
    blocks = [(state.base, state.base_ids, state.base_mask, state.base_active)]
    blocks.extend(zip(state.chunks, state.chunk_ids, state.chunk_masks, state.chunk_active))
    for matrix, ids, mask, active in blocks:
        if not active:
            continue
        scores = exact_scores(matrix, q, reproducible=True)
        scores[~mask] = -np.inf
        selected = top_k_indices(scores, ids, min(k, active), deterministic_ties=True)
        candidates.extend(ids[selected].tolist())
        candidate_scores.extend(scores[selected].tolist())
    if candidates:
        scores = np.asarray(candidate_scores, dtype=np.float32)
        indices = top_k_indices(scores, candidates, min(k, len(state.texts)), deterministic_ties=True)
        result_ids = [candidates[i] for i in indices]
        result_scores = [float(scores[i]) for i in indices]
    else:
        result_ids, result_scores = [], []
    return {"record_ids": result_ids, "scores": result_scores,
            "latency_ms": (time.perf_counter() - start) * 1000, "corpus_version": state.version}


class CorpusSnapshot:
    """One immutable version; old snapshots remain valid after mutations."""

    def __init__(self, state, embedding_space):
        self._state = state
        self.embedding_space = embedding_space

    @property
    def version(self):
        return self._state.version

    @property
    def active_ids(self):
        return tuple(self._state.texts)

    def record_text(self, rid):
        return self._state.texts[rid]

    def embedding(self, rid):
        return _embedding(self._state, rid).copy()

    def retrieve(self, query_vector, k=5):
        return _search(self._state, query_vector, k)

    def export(self):
        ids, matrix = _materialise(self._state)
        return {rid: self.record_text(rid) for rid in ids}, matrix, ids

    def digest(self):
        """Logical contents only: independent of storage/compaction/version."""
        digest = hashlib.sha256()
        for rid in sorted(self._state.texts):
            for text in (rid, self._state.texts[rid]):
                value = text.encode("utf-8")
                digest.update(len(value).to_bytes(8, "little"))
                digest.update(value)
            digest.update(_embedding(self._state, rid).tobytes())
        return digest.hexdigest()


class DynamicCorpus:
    """Mutations atomically publish a state; readers briefly lock to acquire it.

    Duplicate insertion and missing update/delete are rejected. Batch operations
    are sequential within one all-or-nothing transaction. Supplied embeddings
    must be compatible; dimension/normalisation checks cannot infer model space.
    """

    def __init__(self, records, embeddings, record_ids=None, *, embedding_space="enc_a",
                 compaction=None, normalise_on_write=False):
        if not isinstance(embedding_space, str) or not embedding_space:
            raise ValueError("declare a nonempty embedding_space")
        self.embedding_space = embedding_space
        self.compaction = compaction or CompactionPolicy()
        self.normalise_on_write = normalise_on_write
        self._lock = threading.RLock()
        self._full_rebuild = False
        self._state = self._initial(records, embeddings, record_ids, 0)
        self.dimension = self._state.base.shape[1]

    @staticmethod
    def _initial(records, embeddings, record_ids, version):
        records = dict(records)
        ids = list(records) if record_ids is None else list(record_ids)
        if set(ids) != set(records) or any(not isinstance(value, str) for value in records.values()):
            raise ValueError("records, texts and embedding IDs must match exactly")
        reference = ExactDenseRetriever.from_arrays(embeddings, ids, deterministic_ties=True, reproducible=True)
        return _State(version, _freeze(reference.embeddings), _freeze(np.asarray(ids, dtype=object)),
                      MappingProxyType({rid: i for i, rid in enumerate(ids)}),
                      _freeze(np.ones(len(ids), dtype=bool)), len(ids), (), (), (), (),
                      MappingProxyType({}), MappingProxyType(records))

    def snapshot(self):
        with self._lock:
            return CorpusSnapshot(self._state, self.embedding_space)

    def record_text(self, record_id):
        return self.snapshot().record_text(record_id)

    def get_document(self, record_id):
        snapshot = self.snapshot()
        return {"record_id": record_id, "text": snapshot.record_text(record_id),
                "embedding": snapshot.embedding(record_id), "corpus_version": snapshot.version,
                "embedding_space": self.embedding_space}

    def search(self, query_embedding, top_k=5, *, embedding_space=None):
        self._check_space(embedding_space)
        return self.snapshot().retrieve(query_embedding, top_k)

    def retrieve(self, query_vector, k=5):
        return self.search(query_vector, k)

    def _check_space(self, space):
        if space is not None and space != self.embedding_space:
            raise ValueError(f"embedding space mismatch: expected {self.embedding_space}")

    def _vector(self, embedding):
        vector = np.array(embedding, dtype=np.float32, copy=True)
        if vector.ndim != 1 or vector.size != self.dimension or not np.isfinite(vector).all():
            raise ValueError("embedding must be a finite vector of the configured dimension")
        norm = np.linalg.norm(vector)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("embedding has a zero or non-finite norm")
        if self.normalise_on_write:
            vector /= norm
        elif not np.isclose(norm, 1.0, atol=1e-4):
            raise ValueError("embedding must be L2-normalised; normalise explicitly if appropriate")
        return _freeze(vector)

    def _stage(self, state, operations, traces=None):
        texts = dict(state.texts)
        latest = dict(state.delta_latest)
        base_mask = state.base_mask.copy()
        base_active = state.base_active
        masks, active = list(state.chunk_masks), list(state.chunk_active)
        copied_masks = set()
        new_ids, vectors, new_mask = [], [], []
        next_chunk = len(state.chunks)

        def invalidate(rid):
            nonlocal base_active
            base_row = state.base_positions.get(rid)
            if base_row is not None and base_mask[base_row]:
                base_mask[base_row] = False
                base_active -= 1
            old_position = latest.pop(rid, None)
            if old_position is not None:
                chunk, row = old_position
                if chunk == next_chunk:
                    new_mask[row] = False
                else:
                    if chunk not in copied_masks:
                        masks[chunk] = masks[chunk].copy()
                        copied_masks.add(chunk)
                    masks[chunk][row] = False
                    active[chunk] -= 1

        for operation in operations:
            op_start = time.perf_counter() if traces is not None else None
            rid = operation.record_id
            if not isinstance(rid, str) or not rid.strip():
                raise ValueError("record ID must be a nonempty string")
            if operation.kind not in ("insert", "update", "delete"):
                raise ValueError("mutation kind must be insert, update or delete")
            exists = rid in texts
            if operation.kind == "insert" and exists:
                raise KeyError(f"duplicate active record: {rid}")
            if operation.kind in ("update", "delete") and not exists:
                raise KeyError(f"missing active record: {rid}")
            if operation.kind == "delete":
                invalidate(rid)
                del texts[rid]
            else:
                if not isinstance(operation.text, str):
                    raise ValueError("document text must be a string")
                self._check_space(operation.embedding_space)
                vector = self._vector(operation.embedding)
                invalidate(rid)
                texts[rid] = operation.text
                latest[rid] = (next_chunk, len(new_ids))
                new_ids.append(rid)
                vectors.append(vector)
                new_mask.append(True)
            if traces is not None:
                traces.append({"kind": operation.kind, "record_id": rid,
                               "staging_ms": (time.perf_counter() - op_start) * 1000})
        chunks, chunk_ids = state.chunks, state.chunk_ids
        if vectors:
            chunks += (_freeze(np.stack(vectors)),)
            chunk_ids += (_freeze(np.asarray(new_ids, dtype=object)),)
            masks.append(_freeze(np.asarray(new_mask, dtype=bool)))
            active.append(sum(new_mask))
        masks = tuple(_freeze(mask) for mask in masks)
        return _State(state.version + 1, state.base, state.base_ids, state.base_positions,
                      _freeze(base_mask), base_active, chunks, chunk_ids, masks, tuple(active),
                      MappingProxyType(latest), MappingProxyType(texts), state.delta_rows + len(vectors))

    @staticmethod
    def _compact_state(state):
        ids, matrix = _materialise(state)
        # Reuse the existing exact index constructor, including normalisation validation.
        return DynamicCorpus._initial(state.texts, matrix, ids, state.version)

    def _should_compact(self, state):
        if self.compaction.delta_ratio is not None and state.delta_rows / max(1, len(state.texts)) >= self.compaction.delta_ratio:
            return "delta_rows"
        ratio = (len(state.base_ids) - state.base_active) / max(1, len(state.base_ids))
        if self.compaction.tombstone_ratio is not None and ratio >= self.compaction.tombstone_ratio:
            return "base_tombstones"
        return None

    def apply_batch(self, operations: Sequence[Mutation], *, trace_operations=False):
        operations = tuple(operations)
        if any(not isinstance(operation, Mutation) for operation in operations):
            raise TypeError("apply_batch expects Mutation objects")
        start = time.perf_counter()
        with self._lock:
            acquired = time.perf_counter()
            if not operations:
                return {"operations": 0, "version": self._state.version, "mutation_ms": 0.0,
                        "rebuild_ms": 0.0, "compaction_ms": 0.0, "total_ms": 0.0, "compaction_reason": None}
            traces = [] if trace_operations else None
            staged = self._stage(self._state, operations, traces)
            stage_end = time.perf_counter()
            reason = "full_rebuild" if self._full_rebuild else self._should_compact(staged)
            if reason is not None:
                staged = self._compact_state(staged)
            maintenance_end = time.perf_counter()
            self._state = staged  # Only publication point: failed staging/compaction changes nothing.
            end = time.perf_counter()
            maintenance_ms = (maintenance_end - stage_end) * 1000
            return {
                "operations": len(operations), "version": staged.version,
                "lock_wait_ms": (acquired - start) * 1000,
                "mutation_ms": (stage_end - acquired) * 1000,
                "rebuild_ms": maintenance_ms if self._full_rebuild else 0.0,
                "compaction_ms": maintenance_ms if reason and not self._full_rebuild else 0.0,
                "total_ms": (end - start) * 1000, "compaction_reason": reason,
                "active_records": len(staged.texts), "delta_rows": staged.delta_rows,
                "operation_timings": traces,
            }

    def insert_document(self, record_id, text, embedding, *, embedding_space=None):
        return self.apply_batch([Mutation("insert", record_id, text, embedding, embedding_space)])

    def update_document(self, record_id, text, embedding, *, embedding_space=None):
        return self.apply_batch([Mutation("update", record_id, text, embedding, embedding_space)])

    def delete_document(self, record_id):
        return self.apply_batch([Mutation("delete", record_id)])

    def compact(self):
        start = time.perf_counter()
        with self._lock:
            staged = self._compact_state(self._state)
            self._state = replace(staged, version=self._state.version + 1)
            return {"compaction_ms": (time.perf_counter() - start) * 1000,
                    "version": self._state.version}

    def reset(self, records, embeddings, record_ids=None):
        staged = self._initial(records, embeddings, record_ids, 0)
        if staged.base.shape[1] != self.dimension:
            raise ValueError("reset must retain embedding dimension/space")
        with self._lock:
            self._state = replace(staged, version=self._state.version + 1)

    def get_stats(self):
        state = self.snapshot()._state
        base_bytes = state.base.nbytes
        delta_bytes = sum(chunk.nbytes for chunk in state.chunks)
        mask_bytes = state.base_mask.nbytes + sum(mask.nbytes for mask in state.chunk_masks)
        metadata = (sys.getsizeof(dict(state.texts)) + sys.getsizeof(dict(state.delta_latest)) +
                    sys.getsizeof(dict(state.base_positions)) + sys.getsizeof(state.chunks))
        text_bytes = sum(len(text.encode("utf-8")) for text in state.texts.values())
        return {
            "version": state.version, "strategy": "full_rebuild" if self._full_rebuild else "incremental",
            "active_records": len(state.texts), "dimension": self.dimension, "embedding_space": self.embedding_space,
            "base_rows": len(state.base_ids), "base_active_rows": state.base_active,
            "delta_rows": state.delta_rows, "active_delta_rows": len(state.delta_latest),
            "stale_delta_rows": state.delta_rows - len(state.delta_latest), "delta_chunks": len(state.chunks),
            "tombstoned_base_rows": len(state.base_ids) - state.base_active,
            "delta_fraction": state.delta_rows / max(1, len(state.texts)),
            "active_delta_fraction": len(state.delta_latest) / max(1, len(state.texts)),
            "tombstone_fraction": (len(state.base_ids) - state.base_active) / max(1, len(state.base_ids)),
            "base_matrix_bytes": base_bytes, "delta_matrix_bytes": delta_bytes,
            "embedding_bytes": base_bytes + delta_bytes, "mask_bytes": mask_bytes,
            "id_pointer_array_bytes": state.base_ids.nbytes + sum(ids.nbytes for ids in state.chunk_ids),
            "text_payload_utf8_bytes": text_bytes, "python_metadata_shallow_bytes": metadata,
            "memory_note": "Current state only; array bytes exact, metadata shallow; excludes old reader-held snapshots and RSS",
        }


class FullRebuildCorpus(DynamicCorpus):
    """Control: validate one atomic batch, then rebuild all active vectors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._full_rebuild = True
