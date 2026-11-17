"""Immutable source catalogs and replayable mutations referencing compatible vectors."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np

from src.data import CourseData
from src.retriever import ExactDenseRetriever, normalise_query
from .index import Mutation
from .io import fingerprint


@dataclass
class Catalog:
    records: dict
    vectors: np.ndarray
    ids: list[str]
    embedding_space: str
    source: str

    def __post_init__(self):
        if set(self.ids) != set(self.records):
            raise ValueError("catalog text/ID mapping mismatch")
        checked = ExactDenseRetriever.from_arrays(self.vectors, self.ids)
        self.vectors = checked.embeddings
        self.vectors.setflags(write=False)
        self.records = dict(self.records)
        self.ids = list(self.ids)
        self.positions = {rid: i for i, rid in enumerate(self.ids)}

    @classmethod
    def course(cls, data_dir, space="enc_a"):
        data_dir = Path(data_dir)
        required = [data_dir/"corpus.jsonl", data_dir/f"embeddings/corpus_{space}.npy",
                    data_dir/f"embeddings/corpus_{space}_ids.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ValueError("Missing immutable course assets: " + ", ".join(missing))
        data = CourseData(data_dir)
        ids = json.loads(required[2].read_text())
        # Explicit ID lookup, never align vectors with corpus JSONL row order.
        if any(rid not in data.records for rid in ids):
            raise ValueError("embedding ID missing from corpus")
        return cls({rid: data.record_text(rid) for rid in ids}, np.load(required[1], mmap_mode="r"),
                   ids, space, f"course_precomputed_{space}")

    @classmethod
    def synthetic(cls, count=24000, dimension=64, seed=2026):
        if count < 2 or dimension < 1:
            raise ValueError("synthetic catalog needs count>=2 and positive dimension")
        rng = np.random.default_rng(seed)
        vectors = rng.normal(size=(count, dimension)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        ids = [f"fixture-{i:06d}" for i in range(count)]
        return cls({rid: f"Synthetic source {rid}; a controlled vector fixture, not course evidence." for rid in ids},
                   vectors, ids, f"synthetic_{dimension}", "synthetic_normalised_vectors")

    def subset(self, count, seed=2026):
        if not 1 <= count <= len(self.ids):
            raise ValueError("subset size exceeds available catalog")
        indices = np.random.default_rng(seed).permutation(len(self.ids))[:count]
        ids = [self.ids[i] for i in indices]
        return Catalog({rid: self.records[rid] for rid in ids}, self.vectors[indices], ids,
                       self.embedding_space, self.source)

    def vector(self, rid):
        return self.vectors[self.positions[rid]]

    def fingerprint(self):
        digest = hashlib.sha256()
        digest.update(self.embedding_space.encode())
        for rid in self.ids:
            for value in (rid, self.records[rid]):
                raw = value.encode()
                digest.update(len(raw).to_bytes(8, "little"))
                digest.update(raw)
            digest.update(self.vector(rid).tobytes())
        return digest.hexdigest()

    def initial_ids(self, fraction=0.8, seed=2026):
        if not 0 < fraction < 1:
            raise ValueError("initial fraction must be between zero and one to reserve an update pool")
        count = max(1, min(len(self.ids) - 1, int(len(self.ids) * fraction)))
        return sorted(random.Random(seed).sample(self.ids, count))


@dataclass
class Queries:
    ids: list[str]
    vectors: np.ndarray
    source: str
    embedding_space: str
    transcripts: list[str] | None = None
    audio_files: list[str] | None = None
    split: str | None = None

    def __post_init__(self):
        self.vectors = np.asarray(self.vectors, dtype=np.float32)
        if self.vectors.ndim != 2 or len(self.ids) != len(self.vectors) or not len(self.ids):
            raise ValueError("query IDs/vectors must be nonempty and aligned")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate query IDs")
        for vector in self.vectors:
            normalise_query(vector, self.vectors.shape[1])
        for values in (self.transcripts, self.audio_files):
            if values is not None and len(values) != len(self.ids):
                raise ValueError("query metadata lengths do not match")

    @classmethod
    def probes(cls, catalog, count=4, seed=2026):
        if count < 1:
            raise ValueError("probe count must be positive")
        chosen = np.random.default_rng(seed).choice(len(catalog.ids), size=min(count, len(catalog.ids)), replace=False)
        return cls([f"vector-probe-{i}" for i in range(len(chosen))], catalog.vectors[chosen].copy(),
                   catalog.source + "_vector_probes_NOT_judged_queries", catalog.embedding_space)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as arrays:
            vectors = arrays["vectors"].copy()
            ids = arrays["query_ids"].tolist()
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        return cls(ids, vectors, meta["source"], meta["embedding_space"],
                   meta.get("transcripts"), meta.get("audio_files"), meta.get("split"))


def make_mutation_log(catalog, initial_ids, count, profile="mixed", update_mode="refresh", seed=2026):
    """Mixed is exactly 4 insert / 3 update / 3 delete per complete ten ops.

    Replacements copy a documented source text/vector pair under a logical target
    ID. They invalidate original qrels for that target; refresh preserves content.
    Source assets and their own ID/vector mapping are never rewritten.
    """
    if count < 1 or profile not in ("insert", "update", "delete", "mixed") or update_mode not in ("refresh", "replace"):
        raise ValueError("invalid mutation workload")
    rng = random.Random(seed)
    plan = (["insert"] * 4 + ["update"] * 3 + ["delete"] * 3) * ((count + 9) // 10)
    plan = plan[:count] if profile == "mixed" else [profile] * count
    rng.shuffle(plan)
    active = sorted(initial_ids)
    inactive = sorted(set(catalog.ids) - set(active))

    def move(source, target, position):
        value = source[position]
        source[position] = source[-1]
        source.pop()
        target.append(value)
        return value

    log = []
    for i in range(count):
        kind = plan[i]
        blocked = (kind == "insert" and not inactive) or (kind in ("update", "delete") and not active)
        if blocked:
            needed = "delete" if kind == "insert" else "insert"
            replacement = next((j for j in range(i + 1, count) if plan[j] == needed), None)
            if replacement is None:
                raise ValueError("workload exhausts the active/update pool; reduce operations or adjust initial fraction")
            plan[i], plan[replacement] = plan[replacement], plan[i]
            kind = plan[i]
        if kind == "insert":
            rid = move(inactive, active, rng.randrange(len(inactive)))
            source_id = rid
        elif kind == "delete":
            rid = move(active, inactive, rng.randrange(len(active)))
            source_id = None
        else:
            rid = rng.choice(active)
            source_id = rid if update_mode == "refresh" else rng.choice(catalog.ids)
        log.append({"operation": i, "kind": kind, "record_id": rid, "source_record_id": source_id})
    return log


def mutation_from_entry(catalog, entry):
    if entry["kind"] == "delete":
        return Mutation("delete", entry["record_id"])
    source = entry["source_record_id"]
    return Mutation(entry["kind"], entry["record_id"], catalog.records[source], catalog.vector(source), catalog.embedding_space)


def apply_oracle(oracle, sources, catalog, entry):
    """Independent logical map for validation; no dynamic-index state is used."""
    rid, kind = entry["record_id"], entry["kind"]
    if kind == "insert" and rid in oracle or kind in ("update", "delete") and rid not in oracle:
        raise ValueError("invalid replay operation")
    if kind == "delete":
        del oracle[rid]
        del sources[rid]
    else:
        source = entry["source_record_id"]
        oracle[rid] = (catalog.records[source], catalog.vector(source))
        sources[rid] = source


def materialise_oracle(oracle):
    ids = sorted(oracle)
    if not ids:
        raise ValueError("use the configured dimension to construct an empty reference matrix")
    return ids, np.stack([oracle[rid][1] for rid in ids]), {rid: oracle[rid][0] for rid in ids}
