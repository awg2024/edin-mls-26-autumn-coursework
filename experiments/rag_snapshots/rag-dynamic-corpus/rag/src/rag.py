from __future__ import annotations

import time


class RAGPipeline:
    """
    Main pipeline
    """
    def __init__(
        self,
        asr,
        encoder,
        retriever,
        data,
        context_builder,
        generator,
        reranker=None,
    ):
        self.asr = asr
        self.encoder = encoder
        self.retriever = retriever
        self.data = data
        self.context_builder = context_builder
        self.generator = generator
        self.reranker = reranker

    def answer_audio(self, audio_path, k: int = 5, final_k: int = 5):
        total_start = time.perf_counter()

        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio_path)
        asr_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        query_vector = self.encoder.encode(transcript)
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        retrieval = self.retriever.retrieve(query_vector, k=k)

        final_retrieval = retrieval  # Optional cross-encoder reranking
        rerank_ms = 0.0

        if self.reranker is not None:
            t0 = time.perf_counter()

            final_retrieval = self.reranker.rerank(
                query=transcript, retrieval=retrieval, data=self.data, final_k=final_k)

            rerank_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()

        built = self.context_builder.build(transcript, final_retrieval["record_ids"], self.data)

        context_ms = (time.perf_counter() - t0) * 1000.0
                
        generation = self.generator.generate(built["prompt"])
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return {
            "prompt": built["prompt"],
            "selected_evidence": built.get("selected_evidence", []),
            "transcript": transcript,
            "retrieval": retrieval,
            "final_retrieval": final_retrieval,
            "contexts": built["contexts"],
            "answer": generation["answer"],
            "timing": {
                "asr_ms": asr_ms,
                "embedding_ms": embedding_ms,
                "retrieval_ms": retrieval["latency_ms"],
                "generation_ms": generation["latency_ms"],
                "context_ms": context_ms,
                "end_to_end_ms": total_ms,
                "rerank_ms": rerank_ms
            },
        }
