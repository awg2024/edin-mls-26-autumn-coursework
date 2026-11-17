"""Transactional mutation, exact search and snapshot consistency tests."""
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retriever import ExactDenseRetriever
from extensions.dynamic_corpus import CompactionPolicy, DynamicCorpus, FullRebuildCorpus, Mutation


def fixture(cls=DynamicCorpus, **kwargs):
    return cls({"a": "Alpha text", "b": "Beta text", "c": "Gamma text"}, np.eye(3, dtype=np.float32), **kwargs)


class MutationTests(unittest.TestCase):
    def test_insert_search_and_text(self):
        index = fixture()
        index.insert_document("d", "Delta text", np.array([1, 1, 1], np.float32)/np.sqrt(3))
        self.assertIn("d", index.retrieve([1, 1, 1], 1)["record_ids"])
        self.assertEqual(index.record_text("d"), "Delta text")

    def test_duplicate_insert_and_missing_updates_deletes_rejected(self):
        index = fixture()
        with self.assertRaises(KeyError):
            index.insert_document("a", "duplicate", [1, 0, 0])
        with self.assertRaises(KeyError):
            index.update_document("absent", "text", [1, 0, 0])
        with self.assertRaises(KeyError):
            index.delete_document("absent")

    def test_update_replaces_text_vector_and_masks_old_base(self):
        index = fixture()
        index.update_document("a", "Newest text", [0, 1, 0])
        self.assertEqual(index.record_text("a"), "Newest text")
        np.testing.assert_array_equal(index.get_document("a")["embedding"], [0, 1, 0])
        self.assertEqual(index.search([1, 0, 0], 3)["scores"], [0, 0, 0])
        self.assertEqual(index.get_stats()["tombstoned_base_rows"], 1)

    def test_repeated_update_has_one_active_id_and_no_old_vectors(self):
        index = fixture()
        for i in range(10):
            index.update_document("a", f"version {i}", [0, 1, 0])
        self.assertEqual(index.search([0, 1, 0], 10)["record_ids"].count("a"), 1)
        self.assertEqual(index.record_text("a"), "version 9")
        self.assertEqual(index.get_stats()["active_delta_rows"], 1)
        self.assertEqual(index.get_stats()["stale_delta_rows"], 9)

    def test_delete_never_returned_and_reinsert_after_delete(self):
        index = fixture()
        index.delete_document("a")
        self.assertNotIn("a", index.search([1, 0, 0], 100)["record_ids"])
        with self.assertRaises(KeyError):
            index.record_text("a")
        index.insert_document("a", "Reinserted", [0, 0, 1])
        self.assertEqual(index.record_text("a"), "Reinserted")
        self.assertEqual(index.get_stats()["active_records"], 3)

    def test_delete_delta_row_and_update_then_delete_inside_batch(self):
        index = fixture()
        index.apply_batch([Mutation("insert", "d", "text", [1, 0, 0]), Mutation("delete", "d"),
                           Mutation("update", "a", "new", [0, 1, 0]), Mutation("delete", "a")])
        self.assertEqual(set(index.search([1, 0, 0], 10)["record_ids"]), {"b", "c"})

    def test_invalid_vectors_and_spaces_rejected_without_mutation(self):
        index = fixture()
        original = index.snapshot().digest()
        for vector in ([1, 0], [0, 0, 0], [float("nan"), 0, 1], [float("inf"), 0, 0], [2, 0, 0], [[1, 0, 0]]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                index.insert_document("bad", "text", vector)
            self.assertEqual(index.snapshot().digest(), original)
        with self.assertRaises(ValueError):
            index.insert_document("bad", "text", [1, 0, 0], embedding_space="enc_b")
        with self.assertRaises(ValueError):
            index.search([1, 0, 0], embedding_space="enc_b")

    def test_explicit_normalisation_on_write(self):
        index = fixture(normalise_on_write=True)
        index.insert_document("d", "text", [3, 4, 0])
        np.testing.assert_allclose(index.get_document("d")["embedding"], [0.6, 0.8, 0], atol=1e-7)

    def test_inputs_are_copied_and_snapshots_cannot_be_mutated_via_vector_copy(self):
        matrix = np.eye(2, dtype=np.float32)
        records = {"a": "original", "b": "second"}
        index = DynamicCorpus(records, matrix)
        matrix[0] = [0, 1]
        records["a"] = "external mutation"
        self.assertEqual(index.record_text("a"), "original")
        vector = np.array([1, 0], np.float32)
        index.insert_document("c", "new", vector)
        vector[:] = 0
        value = index.snapshot().embedding("c")
        value[:] = 0
        np.testing.assert_array_equal(index.get_document("c")["embedding"], [1, 0])

    def test_batch_is_atomic_on_late_failure(self):
        index = fixture()
        index.update_document("a", "already delta", [0, 1, 0])
        before, version = index.snapshot().digest(), index.get_stats()["version"]
        with self.assertRaises(KeyError):
            index.apply_batch([Mutation("update", "a", "should roll back", [0, 0, 1]),
                               Mutation("insert", "d", "also roll back", [1, 0, 0]),
                               Mutation("delete", "missing")])
        self.assertEqual(index.snapshot().digest(), before)
        self.assertEqual(index.get_stats()["version"], version)

    def test_compaction_failure_does_not_publish_partial_batch(self):
        index = fixture(compaction=CompactionPolicy(delta_ratio=0.01))
        before = index.snapshot().digest()
        with patch.object(index, "_compact_state", side_effect=MemoryError("fixture")):
            with self.assertRaises(MemoryError):
                index.update_document("a", "rollback", [0, 1, 0])
        self.assertEqual(index.snapshot().digest(), before)

    def test_empty_corpus_and_k_larger_than_active(self):
        index = DynamicCorpus({}, np.empty((0, 3), np.float32))
        self.assertEqual(index.search([1, 0, 0], 10)["record_ids"], [])
        index.insert_document("a", "first", [1, 0, 0])
        self.assertEqual(index.search([1, 0, 0], 100)["record_ids"], ["a"])
        index.delete_document("a")
        index.compact()
        self.assertEqual(index.search([1, 0, 0], 5)["scores"], [])

    def test_invalid_query_and_top_k(self):
        index = fixture()
        for query in ([1, 0], [0, 0, 0], [float("nan"), 0, 1], [float("inf"), 0, 1]):
            with self.assertRaises(ValueError):
                index.search(query)
        for k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                index.search([1, 0, 0], k)

    def test_reset_reconstructs_original_state(self):
        index = fixture()
        original = index.snapshot().digest()
        index.delete_document("a")
        index.reset({"a": "Alpha text", "b": "Beta text", "c": "Gamma text"}, np.eye(3, dtype=np.float32))
        self.assertEqual(index.snapshot().digest(), original)
        with self.assertRaises(ValueError):
            index.reset({"a": "text"}, np.eye(1, dtype=np.float32))

    def test_malformed_corpus_and_policy_rejected(self):
        for ids in (["a", "a"], ["a", "missing"]):
            with self.assertRaises(ValueError):
                DynamicCorpus({"a": "A", "b": "B"}, np.eye(2, dtype=np.float32), ids)
        with self.assertRaises(ValueError):
            CompactionPolicy(delta_ratio=0)
        with self.assertRaises(ValueError):
            CompactionPolicy(tombstone_ratio=float("nan"))


class ExactnessTests(unittest.TestCase):
    def test_global_top_k_across_base_and_multiple_chunks(self):
        index = fixture()
        index.insert_document("z", "one", [1, 0, 0])
        index.insert_document("d", "two", np.array([1, 1, 0], np.float32)/np.sqrt(2))
        self.assertEqual(index.search([1, 0, 0], 3)["record_ids"], ["a", "z", "d"])

    def test_score_ties_use_ascending_ids_across_storage_partitions(self):
        index = DynamicCorpus({"z": "last", "m": "middle"}, np.array([[1, 0], [1, 0]], np.float32))
        index.insert_document("a", "first", [1, 0])
        self.assertEqual(index.search([1, 0], 2)["record_ids"], ["a", "m"])
        index.compact()
        self.assertEqual(index.search([1, 0], 2)["record_ids"], ["a", "m"])

    def test_manual_and_automatic_compaction_preserve_queries(self):
        index = fixture()
        index.update_document("a", "new", [0, 1, 0])
        before = index.search([0, 1, 0], 3)
        index.compact()
        after = index.search([0, 1, 0], 3)
        self.assertEqual(before["record_ids"], after["record_ids"])
        self.assertEqual(before["scores"], after["scores"])
        self.assertEqual(index.get_stats()["delta_rows"], 0)
        auto = fixture(compaction=CompactionPolicy(delta_ratio=0.2))
        receipt = auto.update_document("a", "new", [0, 1, 0])
        self.assertEqual(receipt["compaction_reason"], "delta_rows")
        self.assertEqual(auto.get_stats()["delta_rows"], 0)

    def test_tombstone_threshold_and_stale_delta_threshold(self):
        index = fixture(compaction=CompactionPolicy(tombstone_ratio=0.2))
        self.assertEqual(index.delete_document("a")["compaction_reason"], "base_tombstones")
        self.assertEqual(index.get_stats()["base_rows"], 2)

    def test_random_mutation_sequence_agrees_with_independent_logical_oracle(self):
        rng = np.random.default_rng(31)
        matrix = rng.normal(size=(25, 16)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        records = {f"d{i}": f"text {i}" for i in range(25)}
        ids = list(records)
        dynamic = DynamicCorpus(records, matrix, ids)
        rebuilt = FullRebuildCorpus(records, matrix, ids)
        oracle = {rid: (records[rid], matrix[i].copy()) for i, rid in enumerate(ids)}
        for step in range(30):
            if step % 3 == 0:
                rid, vector = f"new-{step}", matrix[(step + 7) % 25]
                operation = Mutation("insert", rid, f"inserted {step}", vector)
                oracle[rid] = (operation.text, vector)
            elif step % 3 == 1:
                rid = sorted(oracle)[step % len(oracle)]
                vector = matrix[(step + 11) % 25]
                operation = Mutation("update", rid, f"updated {step}", vector)
                oracle[rid] = (operation.text, vector)
            else:
                rid = sorted(oracle)[step % len(oracle)]
                operation = Mutation("delete", rid)
                del oracle[rid]
            dynamic.apply_batch([operation])
            rebuilt.apply_batch([operation])
            if step % 7 == 0:
                dynamic.compact()
            current_ids = sorted(oracle)
            reference = ExactDenseRetriever.from_arrays(np.stack([oracle[r][1] for r in current_ids]), current_ids,
                                                         deterministic_ties=True, reproducible=True)
            for query in matrix[:4]:
                expected = reference.retrieve(query, 10)
                for index in (dynamic, rebuilt):
                    got = index.search(query, 10)
                    self.assertEqual(got["record_ids"], expected["record_ids"])
                    np.testing.assert_allclose(got["scores"], expected["scores"], atol=2e-6, rtol=2e-6)
                    for rid in got["record_ids"]:
                        self.assertEqual(index.record_text(rid), oracle[rid][0])

    def test_static_default_matches_original_blas_and_partition_ranking(self):
        import json
        import tempfile
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(100, 16)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        ids = [f"r{i}" for i in range(100)]
        q = rng.normal(size=16).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            np.save(path/"vectors.npy", matrix)
            (path/"ids.json").write_text(json.dumps(ids))
            legacy = ExactDenseRetriever(path/"vectors.npy", path/"ids.json").retrieve(q, 10)
        scores = matrix @ (q / np.linalg.norm(q))
        selected = np.argpartition(-scores, 9)[:10]
        selected = selected[np.argsort(-scores[selected])]
        self.assertEqual(legacy["record_ids"], [ids[i] for i in selected])
        self.assertEqual(legacy["scores"], [float(scores[i]) for i in selected])
        stable = ExactDenseRetriever.from_arrays(matrix, ids, deterministic_ties=True, reproducible=True).retrieve(q, 10)
        self.assertEqual(legacy["record_ids"], stable["record_ids"])
        np.testing.assert_allclose(legacy["scores"], stable["scores"], atol=2e-6, rtol=2e-6)

    def test_snapshot_remains_consistent_after_completed_mutation(self):
        index = fixture()
        old = index.snapshot()
        index.update_document("a", "latest", [0, 1, 0])
        index.delete_document("b")
        self.assertEqual(old.record_text("a"), "Alpha text")
        self.assertEqual(old.record_text("b"), "Beta text")
        self.assertEqual(index.record_text("a"), "latest")
        self.assertNotIn("b", index.search([0, 1, 0], 3)["record_ids"])

    def test_concurrent_readers_never_observe_partial_atomic_batch(self):
        index = fixture()
        failures = []
        done = threading.Event()
        def reader():
            while not done.is_set():
                snap = index.snapshot()
                for rid in ("a", "b"):
                    if snap.record_text(rid).startswith("version"):
                        other = snap.record_text("b" if rid == "a" else "a")
                        if other != snap.record_text(rid):
                            failures.append((snap.version, rid))
                snap.retrieve([1, 0, 0], 3)
        workers = [threading.Thread(target=reader) for _ in range(2)]
        for worker in workers:
            worker.start()
        for i in range(20):
            index.apply_batch([Mutation("update", "a", f"version {i}", [1, 0, 0]),
                               Mutation("update", "b", f"version {i}", [0, 1, 0])])
        done.set()
        for worker in workers:
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
