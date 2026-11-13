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
        reranker=None, # reranking set to none so we can run baseline 
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

        t0 = time.perf_counter() # start counter 
        transcript = self.asr.transcribe(audio_path) # trasnscribe audio path into text string
        asr_ms = (time.perf_counter() - t0) * 1000.0 # measure latency (ms)

        t0 = time.perf_counter()
        query_vector = self.encoder.encode(transcript) # pass transcript into encoder to create dense embedding vector
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        retrieval = self.retriever.retrieve(query_vector, k=k) # dense retreival of top k closest documents  

        final_retrieval = retrieval  # Optional cross-encoder reranking
        rerank_ms = 0.0

        if self.reranker is not None:  # if reranking is activated ...
            t0 = time.perf_counter() # start reranking build timer 

            final_retrieval = self.reranker.rerank(
                query=transcript, retrieval=retrieval, data=self.data, final_k=final_k)

            rerank_ms = (time.perf_counter() - t0) * 1000.0 # measure in ms 

        t0 = time.perf_counter() # start context selection timer 

        if hasattr(self.context_builder, "build_with_retrieval"):
            built = self.context_builder.build_with_retrieval(transcript, final_retrieval, self.data)
        else:
            built = self.context_builder.build(transcript, final_retrieval["record_ids"], self.data) # context_builder set to simple (baseline) or experimental (with different strategies)

        context_ms = (time.perf_counter() - t0) * 1000.0
                
        generation = self.generator.generate(built["prompt"])
        total_ms = (time.perf_counter() - total_start) * 1000.0

        result = { # return dict structure with information for end user
            "prompt": built["prompt"],
            "selected_evidence": built.get("selected_evidence", []),
            "transcript": transcript,
            "retrieval": retrieval,
            "final_retrieval": final_retrieval,  #  if reranking was enabled you can compare 'retrieval' vs 'final_retrieval' did reranking promote a relevant document? 
            "contexts": built["contexts"],
            "answer": generation["answer"],
            "timing": {
                "asr_ms": asr_ms,
                "embedding_ms": embedding_ms,
                "retrieval_ms": retrieval["latency_ms"],
                "generation_ms": generation["latency_ms"],
                "context_ms": context_ms, # context build time stored to compare against different retrieval stragies 
                "end_to_end_ms": total_ms,
                "rerank_ms": rerank_ms
            },
        }
        if "budget_metrics" in built:
            result["budget_metrics"] = built["budget_metrics"]
            result["timing"]["budget_controller_ms"] = built["budget_metrics"]["budget_controller_ms"]
            result["timing"]["selector_ms"] = built["budget_metrics"]["selector_ms"]
            if "usage" in generation:
                result["generation_usage"] = generation["usage"]
        return result
