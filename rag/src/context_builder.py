from __future__ import annotations


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
            f"Evidence:\n{evidence}"
        )

        return { # return value from the ELM API call 
            "contexts": contexts,
            "prompt": prompt,
        }
