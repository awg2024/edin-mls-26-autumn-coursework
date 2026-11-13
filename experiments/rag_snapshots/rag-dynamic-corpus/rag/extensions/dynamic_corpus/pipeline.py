"""Snapshot-bound adapters around the existing RAGPipeline, without a new generator."""
from __future__ import annotations

import time

from src.rag import RAGPipeline


class _Binding:
    def __init__(self, corpus):
        self.corpus = corpus
        self.snapshot = None
        self.retrieval = None
        self.built = None


class _Retriever:
    def __init__(self, binding):
        self.binding = binding

    def retrieve(self, query_vector, k=5):
        # Acquire at retrieval time, after ASR/encoding, and retain through evidence construction.
        self.binding.snapshot = self.binding.corpus.snapshot()
        result = self.binding.snapshot.retrieve(query_vector, k)
        self.binding.retrieval = result
        return result


class _Data:
    def __init__(self, binding):
        self.binding = binding

    def record_text(self, rid):
        if self.binding.snapshot is None:
            raise RuntimeError("retrieve before reading snapshot evidence")
        return self.binding.snapshot.record_text(rid)

    @property
    def records(self):
        return self.binding.snapshot._state.texts


class _Reranker:
    def __init__(self, binding, reranker):
        self.binding, self.reranker = binding, reranker

    def rerank(self, query, retrieval, data, final_k=5):
        result = self.reranker.rerank(query=query, retrieval=retrieval, data=data, final_k=final_k)
        self.binding.retrieval = result
        return result


class _Builder:
    def __init__(self, binding, builder):
        self.binding, self.builder = binding, builder

    def build(self, transcript, record_ids, data):
        method = getattr(self.builder, "build_with_retrieval", None)
        if callable(method):
            result = method(transcript, {**self.binding.retrieval, "record_ids": record_ids}, data)
        else:
            result = self.builder.build(transcript, record_ids, data)
        self.binding.built = result
        return result


class DynamicRAGPipeline:
    """One binding per request; completed writes are visible to later searches.

    In-flight requests retain their retrieval snapshot even if a later mutation
    deletes/updates a record. Their evidence is explicitly versioned, never a mix
    of new text and old vectors. ASR/generator thread safety remains their own
    responsibility; the index and request bindings do not share mutable readers.
    """

    def __init__(self, asr, encoder, corpus, context_builder, generator, reranker=None):
        self.asr, self.encoder, self.corpus = asr, encoder, corpus
        self.context_builder, self.generator, self.reranker = context_builder, generator, reranker

    def answer_audio(self, audio_path, k=5, final_k=5):
        start = time.perf_counter()
        binding = _Binding(self.corpus)
        pipeline = RAGPipeline(
            self.asr, self.encoder, _Retriever(binding), _Data(binding),
            _Builder(binding, self.context_builder), self.generator,
            reranker=_Reranker(binding, self.reranker) if self.reranker else None,
        )
        result = pipeline.answer_audio(audio_path, k=k, final_k=final_k)
        result["corpus_version"] = binding.snapshot.version
        for key in ("budget_metrics", "discarded_evidence"):
            if key in binding.built:
                result.setdefault(key, binding.built[key])
        result["timing"]["end_to_end_ms"] = (time.perf_counter() - start) * 1000
        return result


def make_pipeline(*, asr, encoder, retriever, data, context_builder, generator, reranker=None, dynamic_corpus=None):
    """Explicit optional backend; disabled mode is the unchanged supplied pipeline."""
    if dynamic_corpus is None:
        return RAGPipeline(asr, encoder, retriever, data, context_builder, generator, reranker=reranker)
    return DynamicRAGPipeline(asr, encoder, dynamic_corpus, context_builder, generator, reranker)
