from __future__ import annotations
from src.context_strategies.tfidf import select_tfidf
from src.context_strategies.mmr import select_mmr
from src.context_strategies.utils import format_evidence

# BASELINE 
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

class ExperimentalContextBuilder:
    def __init__(
        self,
        strategy="tfidf",
        max_chars=5000,
        mmr_lambda=0.7): # hard coded values for context builder. 


        if strategy not in ("tfidf", "mmr"):
            raise ValueError("Unknown context strategy")

        if max_chars < 1:
            raise ValueError("max_chars must be positive")

        self.strategy = strategy
        self.max_chars = max_chars
        self.mmr_lambda = mmr_lambda

    def build(self, transcript, record_ids, data):
        
        documents = [{"record_id": rid, "text": data.record_text(rid)} for rid in record_ids] # extract record id and text from the document 

        # Select the algorithm using an argument.
        if self.strategy == "tfidf": # specified tfidf function call 
            selected = select_tfidf(
                transcript,
                documents,
                max_chars=self.max_chars,
            )

        elif self.strategy == "mmr": # specified mmr function call 
            selected = select_mmr(
                transcript,
                documents,
                max_chars=self.max_chars,
                lambda_=self.mmr_lambda,
            )

        # Construct the evidence string.
        evidence = "\n\n".join(
            format_evidence(item)
            for item in selected
        )

        # Preserve baseline prompt instructions.
        prompt = ( # combine everything into a zero-shot rag system prompt, kept original code the same. 
            "Answer the question using the retrieved evidence below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Question:\n{transcript}\n\n"
            f"Evidence:\n{evidence}"
        )

        return { # return value from the ELM API call 
            "contexts": [
                item["text"] for item in selected
            ],
            "prompt": prompt,
            "selected_evidence": selected,
        }
