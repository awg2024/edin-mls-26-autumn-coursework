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
            f"[Document {i}]\n{text}"
            for i, text in enumerate(contexts, start=1)
        )

        prompt = (
            "Answer the question using the retrieved evidence below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Question:\n{transcript}\n\n"
            f"Evidence:\n{evidence}")

        return {
            "contexts": contexts,
            "prompt": prompt,
        }

class ExperimentalContextBuilder:
    def __init__(
        self,
        strategy="tfidf",
        max_chars=5000,
        mmr_lambda=0.7):


        if strategy not in ("tfidf", "mmr"):
            raise ValueError("Unknown context strategy")

        if max_chars < 1:
            raise ValueError("max_chars must be positive")

        self.strategy = strategy
        self.max_chars = max_chars
        self.mmr_lambda = mmr_lambda

    def build(self, transcript, record_ids, data):
        
        documents = [{"record_id": rid, "text": data.record_text(rid)} for rid in record_ids]

        if self.strategy == "tfidf":
            selected = select_tfidf(
                transcript,
                documents,
                max_chars=self.max_chars,
            )

        elif self.strategy == "mmr":
            selected = select_mmr(
                transcript,
                documents,
                max_chars=self.max_chars,
                lambda_=self.mmr_lambda,
            )

        evidence = "\n\n".join(
            format_evidence(item)
            for item in selected
        )

        # Preserve baseline prompt instructions.
        prompt = (
            "Answer the question using the retrieved evidence below. "
            "If the evidence is insufficient, say so.\n\n"
            f"Question:\n{transcript}\n\n"
            f"Evidence:\n{evidence}"
        )

        return {
            "contexts": [
                item["text"] for item in selected
            ],
            "prompt": prompt,
            "selected_evidence": selected,
        }
