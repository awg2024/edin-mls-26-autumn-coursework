from __future__ import annotations

# REMOVE before testing !!
class SimpleContextBuilder:
    """Intentionally simple baseline; advanced context strategies are left open."""

    def build(self, transcript: str, record_ids: list[str], data) -> dict:
        contexts = [data.record_text(rid) for rid in record_ids]

        evidence = "\n\n".join(
            f"[Document {i}]\n{text}" # indexes each FULL document from start to finish for each (k)
            for i, text in enumerate(contexts, start=1)
        )

        prompt = ( # combine everything into a zero-shot rag system prompt 
            "Answer the question using the retrieved evidence below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Question:\n{transcript}\n\n"
            f"Evidence:\n{evidence}")

        return { # return value from the ELM API call 
            "contexts": contexts,
            "prompt": prompt,
        }

# src/context_builder.py

class ExperimentalContextBuilder:
    def __init__(
        self,
        strategy="tfidf",
        max_tokens=1200,
        mmr_lambda=0.7,
    ):
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.mmr_lambda = mmr_lambda

    def build(self, transcript, record_ids, data):
        
        documents = [{"record_id": rid, "text": data.record_text(rid)} for rid in record_ids] # extract record id and text from the document 

        if self.strategy == "tfidf": # specified tfidf function call 
            selected = select_tfidf(
                transcript,
                documents,
                self.max_tokens)

        elif self.strategy == "mmr": # specified mmr function call 
            selected = select_mmr(
                transcript,
                documents,
                self.max_tokens,
                self.mmr_lambda)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        contexts = [item["text"] for item in selected]

        evidence = "\n\n".join(
            f"[Document {item['record_id']}]\n"
            f"{item['text']}"
            for item in selected)

        prompt = ( # combine everything into a zero-shot rag system prompt, kept original code the same. 
            "Answer the question using the retrieved evidence below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Question:\n{transcript}\n\n"
            f"Evidence:\n{evidence}")

        return {"contexts": contexts, "prompt": prompt} # return value from the ELM API call 