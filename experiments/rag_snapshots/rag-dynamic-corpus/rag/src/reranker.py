# /src/reranker.py 
# reranking script 
import numpy as np

class CrossEncoderReranker:
    """
    Rerank dense-retrieved documents with a cross-encoder.
    """

    def __init__(
        self,
        model_path, # dict containing the pretrained cross-encoder model
        device="cpu", # run the model on CPU unless other device is specified (H200?)
        batch_size=8, # process up to eight query-document pairs together
        max_length=512): # maximum tokenised input sequence lenght , limitation of this reranking process 
        
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            str(model_path),
            device=device,
            max_length=max_length,
            local_files_only=True) # only use loocally available model files 

        self.batch_size = batch_size

    def rerank(self, query, retrieval, data, final_k=5):
        """
        query:
            Original ASR transcript.

        retrieval:
            Dictionary returned by ExactDenseRetriever,
            containing record_ids, scores, latency_ms.

        data:
            CourseData instance for loading document text.

        final_k:
            Number of documents to retain after reranking.
        """
        if final_k < 1:
            raise ValueError("final_k must be positive")

        ids = retrieval["record_ids"]

        if not ids:
            return {"record_ids": [], "scores": [], "dense_scores": []}

        # Create a query-document pair for each candidate.
        pairs = [(query, data.record_text(rid)) for rid in ids]

        # Run the cross-encoder on all candidate pairs.
        scores = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
            dtype=np.float32).reshape(-1)

        # Sort candidate indices by descending reranker score.
        order = np.argsort(-scores, kind="stable")[:final_k]

        return {
            "record_ids": [ids[int(i)] for i in order],
            "scores": [float(scores[i]) for i in order],
            "dense_scores": [retrieval["scores"][int(i)] for i in order]}
