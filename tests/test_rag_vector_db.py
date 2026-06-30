"""Unit tests for the offline RAG vector DB helpers (src/knowledge/rag_vector_db.py).

These cover the pure, dependency-light pieces: the local hash embedding backend,
chunk records, document chunking, source expansion, incremental change detection,
and a build-free build->query round trip against a hand-written DB on disk.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.knowledge.rag_vector_db import (
    PROJECT_ROOT,
    ChunkRecord,
    HashEmbeddingBackend,
    chunk_documents,
    expand_sources,
    get_file_hash,
    identify_new_or_modified_files,
    load_existing_db_state,
    load_vector_db,
    query_vector_db,
    write_json,
)


class HashEmbeddingBackendTests(unittest.TestCase):
    def setUp(self):
        self.backend = HashEmbeddingBackend()

    def test_embed_documents_shape_and_dtype(self):
        matrix = self.backend.embed_documents(["alpha beta", "gamma"])
        self.assertEqual(matrix.shape, (2, 384))
        self.assertEqual(matrix.dtype, np.float32)

    def test_non_empty_rows_are_l2_normalized(self):
        matrix = self.backend.embed_documents(["squat depth knee angle"])
        self.assertAlmostEqual(float(np.linalg.norm(matrix[0])), 1.0, places=5)

    def test_empty_text_is_all_zero_row(self):
        # No tokens -> no division by zero -> a zero vector (not NaN).
        matrix = self.backend.embed_documents([""])
        self.assertEqual(float(np.linalg.norm(matrix[0])), 0.0)
        self.assertFalse(np.isnan(matrix).any())

    def test_embed_query_matches_embed_documents(self):
        text = "knees cave inward"
        np.testing.assert_array_equal(
            self.backend.embed_query(text),
            self.backend.embed_documents([text])[0],
        )

    def test_embedding_is_deterministic(self):
        first = self.backend.embed_query("forward lean torso")
        second = self.backend.embed_query("forward lean torso")
        np.testing.assert_array_equal(first, second)

    def test_dimensions_argument_is_respected(self):
        backend = HashEmbeddingBackend(dimensions=16)
        self.assertEqual(backend.embed_query("hello").shape, (16,))
        self.assertEqual(backend.name, "hash-16")

    def test_identical_text_is_more_similar_than_unrelated_text(self):
        target = self.backend.embed_query("shallow squat depth")
        same = self.backend.embed_query("shallow squat depth")
        other = self.backend.embed_query("banana fruit yellow")
        self.assertGreater(float(target @ same), float(target @ other))

    def test_tokenize_lowercases_and_keeps_words(self):
        tokens = HashEmbeddingBackend._tokenize("Hello World")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)

    def test_hash_to_index_in_range(self):
        for token in ("a", "squat", "knee", "深蹲"):
            index = self.backend._hash_to_index(token)
            self.assertTrue(0 <= index < self.backend.dimensions)


class ChunkRecordTests(unittest.TestCase):
    def test_to_json_round_trips_fields(self):
        record = ChunkRecord(chunk_id="abc", text="hi", metadata={"source": "x.txt"})
        self.assertEqual(
            record.to_json(),
            {"chunk_id": "abc", "text": "hi", "metadata": {"source": "x.txt"}},
        )


class ChunkDocumentsTests(unittest.TestCase):
    def test_enriches_metadata_with_index_and_length(self):
        chunks = chunk_documents(
            [("a short sentence about squats", {"source": "doc.txt", "kind": "text"})],
            chunk_size=900,
            chunk_overlap=0,
        )
        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.metadata["chunk_index"], 0)
        self.assertEqual(chunk.metadata["char_length"], len(chunk.text))
        self.assertEqual(chunk.metadata["source"], "doc.txt")

    def test_blank_documents_produce_no_chunks(self):
        chunks = chunk_documents(
            [("   \n  ", {"source": "blank.txt"})],
            chunk_size=900,
            chunk_overlap=0,
        )
        self.assertEqual(chunks, [])

    def test_chunk_id_is_deterministic(self):
        doc = [("repeatable text body", {"source": "doc.txt", "kind": "text"})]
        first = chunk_documents(doc, chunk_size=900, chunk_overlap=0)
        second = chunk_documents(doc, chunk_size=900, chunk_overlap=0)
        self.assertEqual(first[0].chunk_id, second[0].chunk_id)

    def test_reference_metadata_injects_citation_prefix(self):
        chunks = chunk_documents(
            [("body text", {"source": "doc.txt", "reference": "Smith 2020"})],
            chunk_size=900,
            chunk_overlap=0,
        )
        self.assertTrue(chunks[0].text.startswith("[Source Citation: Smith 2020]"))

    def test_long_text_splits_into_indexed_chunks(self):
        body = ". ".join(f"sentence number {n} about squat mechanics" for n in range(40))
        chunks = chunk_documents(
            [(body, {"source": "doc.txt", "kind": "text"})],
            chunk_size=120,
            chunk_overlap=20,
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            [c.metadata["chunk_index"] for c in chunks],
            list(range(len(chunks))),
        )


class ExpandSourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, name: str, text: str = "x") -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def test_directory_recurses_and_filters_by_suffix(self):
        self._write("a.txt")
        self._write("nested/b.md")
        self._write("c.png")  # unsupported suffix, should be skipped
        found = expand_sources([self.tmp])
        names = sorted(p.name for p in found)
        self.assertEqual(names, ["a.txt", "b.md"])

    def test_nonexistent_source_is_ignored(self):
        self.assertEqual(expand_sources([self.tmp / "missing"]), [])

    def test_explicit_file_is_passed_through(self):
        path = self._write("note.txt")
        self.assertEqual(expand_sources([path]), [path])


class ChangeDetectionTests(unittest.TestCase):
    """get_file_hash / identify_new_or_modified_files use paths relative to PROJECT_ROOT."""

    def setUp(self):
        # Files must live under PROJECT_ROOT so `relative_to(PROJECT_ROOT)` succeeds.
        self.tmp = Path(tempfile.mkdtemp(dir=PROJECT_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def _write(self, name: str, text: str) -> Path:
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_get_file_hash_changes_with_content(self):
        path = self._write("doc.txt", "first")
        original = get_file_hash(path)
        self.assertEqual(get_file_hash(path), original)
        path.write_text("second", encoding="utf-8")
        self.assertNotEqual(get_file_hash(path), original)

    def test_new_and_modified_files_are_selected(self):
        unchanged = self._write("keep.txt", "stable")
        modified = self._write("edit.txt", "before")
        fresh = self._write("new.txt", "brand new")

        rel = lambda p: str(p.relative_to(PROJECT_ROOT))
        states = {
            rel(unchanged): get_file_hash(unchanged),
            rel(modified): get_file_hash(modified),
        }
        modified.write_text("after", encoding="utf-8")

        selected = identify_new_or_modified_files([unchanged, modified, fresh], states)
        self.assertIn(modified, selected)
        self.assertIn(fresh, selected)
        self.assertNotIn(unchanged, selected)

    def test_load_existing_db_state_missing_manifest(self):
        self.assertEqual(load_existing_db_state(self.tmp), {})

    def test_load_existing_db_state_reads_file_hashes(self):
        write_json(self.tmp / "manifest.json", {"file_hashes": {"a.txt": "deadbeef"}})
        self.assertEqual(load_existing_db_state(self.tmp), {"a.txt": "deadbeef"})

    def test_load_existing_db_state_corrupt_manifest(self):
        (self.tmp / "manifest.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(load_existing_db_state(self.tmp), {})


class QueryVectorDbTests(unittest.TestCase):
    """Build a tiny DB on disk by hand, then exercise load_vector_db / query_vector_db."""

    def setUp(self):
        self.db_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.db_dir, ignore_errors=True)

        self.chunks = [
            ChunkRecord("c0", "squat depth knee angle below parallel", {"source": "a"}),
            ChunkRecord("c1", "banana fruit yellow tropical", {"source": "b"}),
        ]
        backend = HashEmbeddingBackend()
        vectors = backend.embed_documents([c.text for c in self.chunks])
        np.save(self.db_dir / "embeddings.npy", vectors)
        write_json(self.db_dir / "chunks.json", [c.to_json() for c in self.chunks])
        write_json(
            self.db_dir / "manifest.json",
            {"dimensions": backend.dimensions, "embedding_backend": backend.name},
        )

    def test_load_vector_db_round_trips(self):
        chunks, vectors, manifest = load_vector_db(self.db_dir)
        self.assertEqual([c.chunk_id for c in chunks], ["c0", "c1"])
        self.assertEqual(vectors.shape, (2, 384))
        self.assertEqual(manifest["dimensions"], 384)

    def test_query_ranks_relevant_chunk_first(self):
        results = query_vector_db("squat depth knee", db_dir=self.db_dir, top_k=5)
        self.assertTrue(results)
        self.assertEqual(results[0]["chunk_id"], "c0")
        self.assertEqual(results[0]["rank"], 1)
        self.assertGreater(results[0]["score"], 0.0)

    def test_top_k_caps_result_count(self):
        results = query_vector_db("squat depth knee", db_dir=self.db_dir, top_k=1)
        self.assertEqual(len(results), 1)

    def test_zero_embedding_query_returns_no_results(self):
        # Empty query -> zero embedding -> every score is ~0 and filtered out.
        self.assertEqual(query_vector_db("", db_dir=self.db_dir, top_k=5), [])


if __name__ == "__main__":
    unittest.main()
