# rag/tests/test_reranker.py

# Real cross-encoder integration test.
# Requires a downloaded model and a CUDA GPU.

import unittest
from pathlib import Path

import torch

from src.reranker import CrossEncoderReranker


class FakeData:
    """Small corpus for testing real neural inference."""

    documents = { # example text documents 
        "D1": "Paris is the capital of France.",  # should be dropped  
        "D2": "Excessive model complexity can cause overfitting.",  # d2 should be selected
        "D3": "Small datasets can increase overfitting.", # d3 should be selectd 
    }

    def record_text(self, record_id):
        return self.documents[record_id]


class TestRealCrossEncoder(unittest.TestCase):

    def test_real_gpu_inference(self):
        model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "cross_encoder") # load cross encoder 

        if not model_path.is_dir(): # dir check 
            self.skipTest("Cross-encoder model not downloaded, please try huggingface download script again. ")

        if not torch.cuda.is_available(): # gpu availability check, this should pass since we are in a compute node 
            self.skipTest("CUDA GPU unavailable")

        reranker = CrossEncoderReranker( # call cross encoder 
            model_path=model_path,
            device="cuda",
            batch_size=3)

        retrieval = {
            "record_ids": ["D1", "D2", "D3"],
            "scores": [0.95, 0.90, 0.85]} # rng numbers from enc_a model, dense cosine similarity scores 

        result = reranker.rerank(
            query="What causes overfitting?",
            retrieval=retrieval,
            data=FakeData(),
            final_k=2) # top-2 relevant results 

        print(f"Real MiniLM reranking result: {result}")

        self.assertEqual(len(result["record_ids"]), 2) # ensure that records and scores are [2, ] as specified. 
        self.assertEqual(len(result["scores"]), 2)

        # Model predictions should be sorted descending.
        self.assertGreaterEqual(result["scores"][0], result["scores"][1])

        self.assertTrue(set(result["record_ids"]).issubset(set(retrieval["record_ids"]))) # ensures that record_ids came from the original retrieval pool 


if __name__ == "__main__":
    unittest.main()
