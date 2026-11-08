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
    ):
        self.asr = asr
        self.encoder = encoder
        self.retriever = retriever
        self.data = data
        self.context_builder = context_builder
        self.generator = generator

    def answer_audio(self, audio_path, k: int = 5):
        total_start = time.perf_counter()

        t0 = time.perf_counter() # start counter 
        transcript = self.asr.transcribe(audio_path) # trasnscribe audio path into text string
        asr_ms = (time.perf_counter() - t0) * 1000.0 # measure latency (ms)

        t0 = time.perf_counter()
        query_vector = self.encoder.encode(transcript) # pass transcript into encoder to create dense embedding vector
        embedding_ms = (time.perf_counter() - t0) * 1000.0

        retrieval = self.retriever.retrieve(query_vector, k=k) # fetch top k closest documents 

       
        t0 = time.perf_counter() # start counter before build 

        built = self.context_builder.build(transcript, retrieval["record_ids"], self.data)
        context_ms = (time.perf_counter() - t0) * 1000.0 # measure build time in ms. 

        generation = self.generator.generate(built["prompt"])
        total_ms = (time.perf_counter() - total_start) * 1000.0

        return { # return dict structure with information for end user 
            "transcript": transcript,
            "retrieval": retrieval,
            "contexts": built["contexts"],
            "answer": generation["answer"],
            "timing": {
                "asr_ms": asr_ms,
                "embedding_ms": embedding_ms,
                "retrieval_ms": retrieval["latency_ms"],
                "generation_ms": generation["latency_ms"],
                "context_ms": context_ms, # context build time stored to compare against different retrieval stragies 
                "end_to_end_ms": total_ms,
            },
        }
