"""Tests for exact vector search (app/search/): maths, ranking, candidates, service.

Ranking tests use hand-made unit vectors, NOT a model: the expected order is
known exactly, so the tests fail if the similarity maths or the sorting
break. The real model is only used in the opt-in test_search_smoke.py.
Temporary folders and databases only; no .env, no network.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import hashlib
import logging
import math
import sqlite3
import struct
import unittest
from contextlib import closing

import numpy as np

from app.document_processing import database
from app.document_processing.chunking import ChunkingConfig
from app.document_processing.models import (
    DocumentNotFoundError,
    InvalidDocumentIdError,
    ProcessingError,
    StorageError,
)
from app.embeddings import storage as embedding_storage
from app.embeddings.models import NewEmbedding
from app.embeddings.vectors import l2_normalize, text_sha256, to_float32
from app.search import service, storage
from app.search.models import (
    DEFAULT_TOP_K,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    InvalidQueryVectorError,
    InvalidSearchQueryError,
    InvalidTopKError,
    SearchNotSupportedError,
    validate_query,
    validate_top_k,
)
from app.search.similarity import (
    VectorValidationError,
    as_candidate_matrix,
    as_query_vector,
    cosine_similarities,
)
from fake_embeddings import FakeEmbeddingProvider
from retrieval_baseline import recall_at_k
from test_embedding_storage import EmbeddingTestCase

logging.getLogger("app").setLevel(logging.CRITICAL)

DIMENSION = 8  # FakeEmbeddingProvider's default


def unit(*values, dimension=DIMENSION):
    """A float32 unit vector from its first components (the rest are 0)."""
    padded = list(values) + [0.0] * (dimension - len(values))
    return to_float32(l2_normalize(padded))


QUERY_VECTOR = unit(1.0)  # the direction every test query points to
# Expected cosine similarities with QUERY_VECTOR (computed by hand):
VECTOR_A = unit(0.9, 0.1)  # 0.9 / sqrt(0.82) = 0.99388  -> rank 1
VECTOR_B = unit(0.6, 0.8)  # 0.6                        -> rank 2
VECTOR_C = unit(0.0, 0.0, 1.0)  # 0.0 (unrelated)      -> rank 3
VECTOR_D = unit(-1.0)  # -1.0 (opposite)               -> rank 4


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


class ScriptedProvider(FakeEmbeddingProvider):
    """FakeEmbeddingProvider whose query vector is chosen by the test.

    The contract, the "query: " prefix and model_inputs work exactly like
    the fake provider; only the returned query vector is replaced.
    """

    def __init__(self, query_vector=QUERY_VECTOR, **options):
        super().__init__(**options)
        self.query_vector = query_vector

    def embed_query(self, text):
        super().embed_query(text)  # records "query: " + text in model_inputs
        return self.query_vector


class SearchTestCase(EmbeddingTestCase):
    """Documents with one chunk each, and hand-made vectors stored for them."""

    def setUp(self):
        super().setUp()
        self.provider = ScriptedProvider()

    def one_chunk_document(self, text):
        document_id = self.chunked_document(text.encode("utf-8"), ChunkingConfig(1200, 200))
        [chunk] = self.chunks(document_id)
        return chunk

    def store_vector(self, chunk, vector, contract=None, truncated=False):
        contract = contract or self.provider.contract
        embedding_storage.save_embeddings(self.db_path, contract, [NewEmbedding(
            chunk_id=chunk.chunk_id,
            text_sha256=text_sha256(chunk.text),
            input_sha256=text_sha256(contract.passage_input(chunk.text)),
            vector=vector,
            truncated=truncated,
        )])

    def corpus(self):
        """Four chunks (A, B, C, D) in four documents, with the vectors above."""
        chunks = {}
        for name, vector in [("A", VECTOR_A), ("B", VECTOR_B), ("C", VECTOR_C),
                             ("D", VECTOR_D)]:
            chunks[name] = self.one_chunk_document(f"Chunk {name}: some text about {name}.")
            self.store_vector(chunks[name], vector)
        return chunks

    def run_search(self, query="What is FastAPI?", provider=None, **options):
        return service.search(query, db_path=self.db_path, provider=provider or self.provider,
                              **options)

    def result_ids(self, results):
        """chunk_ids in rank order, from SearchResults or its .results tuple."""
        items = results.results if hasattr(results, "results") else results
        return [result.chunk.chunk_id for result in items]

    def execute(self, sql, parameters=(), ignore_checks=False):
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            if ignore_checks:
                connection.execute("PRAGMA ignore_check_constraints = ON")
            connection.execute(sql, parameters)

    def database_snapshot(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return {table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                    for table in ("documents", "chunks", "embeddings")}


# --- Similarity maths (no database) -------------------------------------------


class SimilarityTests(unittest.TestCase):
    def test_scores_are_the_dot_products_of_unit_vectors(self):
        matrix = as_candidate_matrix([VECTOR_A, VECTOR_B, VECTOR_C, VECTOR_D], DIMENSION)
        query = as_query_vector(QUERY_VECTOR, DIMENSION)

        scores = cosine_similarities(matrix, query)

        expected = [dot(QUERY_VECTOR, v) for v in (VECTOR_A, VECTOR_B, VECTOR_C, VECTOR_D)]
        np.testing.assert_allclose(scores, expected, atol=1e-7)
        np.testing.assert_allclose(scores, [0.9 / math.sqrt(0.82), 0.6, 0.0, -1.0], atol=1e-6)

    def test_scores_equal_the_cosine_formula(self):
        a, b = unit(1.0, 2.0, 3.0), unit(-2.0, 0.5, 1.0)
        cosine = dot(a, b) / (math.sqrt(dot(a, a)) * math.sqrt(dot(b, b)))

        [score] = cosine_similarities(as_candidate_matrix([b], DIMENSION),
                                      as_query_vector(a, DIMENSION))

        self.assertAlmostEqual(score, cosine, places=6)

    def test_identical_vectors_score_one(self):
        [score] = cosine_similarities(as_candidate_matrix([VECTOR_B], DIMENSION),
                                      as_query_vector(VECTOR_B, DIMENSION))

        self.assertAlmostEqual(score, 1.0, places=6)

    def test_query_vector_with_wrong_dimension_is_rejected(self):
        for bad in [unit(1.0, dimension=4), unit(1.0, dimension=9), (), ((1.0,),)]:
            with self.subTest(length=len(bad)), self.assertRaises(VectorValidationError):
                as_query_vector(bad, DIMENSION)

    def test_non_finite_query_vector_is_rejected(self):
        for bad_value in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=bad_value), self.assertRaises(VectorValidationError):
                as_query_vector((bad_value,) + (0.0,) * 7, DIMENSION)

    def test_query_vector_that_is_not_normalized_is_rejected(self):
        for bad in [(2.0,) + (0.0,) * 7, (0.0,) * 8, (0.5,) + (0.0,) * 7]:
            with self.subTest(vector=bad), self.assertRaises(VectorValidationError):
                as_query_vector(bad, DIMENSION)

    def test_non_numeric_query_vector_is_rejected(self):
        with self.assertRaises(VectorValidationError):
            as_query_vector(("x",) * 8, DIMENSION)

    def test_candidate_with_wrong_dimension_is_rejected(self):
        for bad in [[unit(1.0, dimension=4)], [VECTOR_A, unit(1.0, dimension=9)], []]:
            with self.subTest(rows=len(bad)), self.assertRaises(VectorValidationError):
                as_candidate_matrix(bad, DIMENSION)

    def test_non_finite_or_non_unit_candidates_are_rejected_not_renormalized(self):
        bad_rows = [(float("nan"),) + (0.0,) * 7, (float("inf"),) + (0.0,) * 7,
                    (3.0,) + (0.0,) * 7, (0.0,) * 8]
        for bad in bad_rows:
            with self.subTest(vector=bad), self.assertRaises(VectorValidationError) as caught:
                as_candidate_matrix([VECTOR_A, bad], DIMENSION)
            self.assertIn("Candidate vector 1", str(caught.exception))

    def test_float32_rounding_of_a_unit_vector_is_accepted(self):
        values = to_float32(l2_normalize([0.123456789 * (i + 1) for i in range(384)]))

        matrix = as_candidate_matrix([values], 384)

        self.assertEqual(matrix.shape, (1, 384))

    def test_matrix_and_query_of_different_sizes_are_rejected(self):
        with self.assertRaises(VectorValidationError):
            cosine_similarities(np.ones((2, 4)) / 2, as_query_vector(QUERY_VECTOR, DIMENSION))

    def test_scores_outside_minus_one_to_one_are_rejected(self):
        # Bypasses the checks on purpose: the invariant must still hold.
        with self.assertRaises(VectorValidationError):
            cosine_similarities(np.full((1, DIMENSION), 2.0), np.full(DIMENSION, 2.0))
        with self.assertRaises(VectorValidationError):
            cosine_similarities(np.full((1, DIMENSION), np.nan), np.ones(DIMENSION))


# --- Ranking (pure function) --------------------------------------------------


class RankingTests(unittest.TestCase):
    def test_highest_score_comes_first(self):
        order = service.rank_top_k([0.1, 0.9, -0.5, 0.4], ["a", "b", "c", "d"], top_k=4)

        self.assertEqual(order, [1, 3, 0, 2])

    def test_equal_scores_are_ordered_by_chunk_id(self):
        order = service.rank_top_k([0.5, 0.5, 0.5, 0.7], ["c", "a", "b", "z"], top_k=4)

        self.assertEqual(order, [3, 1, 2, 0])

    def test_input_order_does_not_change_the_ranking(self):
        scores = {"x1": 0.3, "x2": 0.3, "x3": 0.8, "x4": -0.1, "x5": 0.3}
        expected = ["x3", "x1", "x2", "x5", "x4"]
        for ids in [list(scores), list(reversed(scores)), ["x5", "x3", "x1", "x4", "x2"]]:
            with self.subTest(order=ids):
                order = service.rank_top_k([scores[i] for i in ids], ids, top_k=5)
                self.assertEqual([ids[i] for i in order], expected)

    def test_float32_noise_counts_as_a_tie(self):
        order = service.rank_top_k([0.5 + 1e-9, 0.5], ["b", "a"], top_k=2)

        self.assertEqual(order, [1, 0])

    def test_only_top_k_are_returned(self):
        self.assertEqual(service.rank_top_k([0.1, 0.9, 0.5], ["a", "b", "c"], top_k=2), [1, 2])
        self.assertEqual(service.rank_top_k([0.1], ["a"], top_k=5), [0])
        self.assertEqual(service.rank_top_k([], [], top_k=5), [])

    def test_scores_and_ids_must_match(self):
        with self.assertRaises(ValueError):
            service.rank_top_k([0.1, 0.2], ["a"], top_k=2)


# --- Validation (pure functions) ----------------------------------------------


class ValidationTests(unittest.TestCase):
    def test_query_is_trimmed(self):
        self.assertEqual(validate_query("  What is FastAPI?\n"), "What is FastAPI?")
        self.assertEqual(validate_query("FastAPI 是做什么的？"), "FastAPI 是做什么的？")

    def test_empty_blank_long_or_non_text_queries_are_rejected(self):
        for bad in ["", " ", "\n\t  ", "x" * (MAX_QUERY_LENGTH + 1), None, 42, b"query"]:
            with self.subTest(query=repr(bad)[:20]), self.assertRaises(InvalidSearchQueryError):
                validate_query(bad)

    def test_longest_allowed_query_is_accepted(self):
        self.assertEqual(len(validate_query("x" * MAX_QUERY_LENGTH)), MAX_QUERY_LENGTH)

    def test_top_k_limits(self):
        self.assertEqual((DEFAULT_TOP_K, MAX_TOP_K), (5, 50))
        for good in [1, 5, MAX_TOP_K]:
            self.assertEqual(validate_top_k(good), good)
        for bad in [0, -1, MAX_TOP_K + 1, 1_000_000, True, 5.0, "5", None]:
            with self.subTest(top_k=bad), self.assertRaises(InvalidTopKError):
                validate_top_k(bad)


class RecallAtKTests(unittest.TestCase):
    """The retrieval-baseline metric itself (the baseline runs in test_search_smoke.py)."""

    def test_recall_at_k(self):
        ranked = {"q1": ["a", "b", "c"], "q2": ["x", "y", "z"], "q3": ["m", "n", "o"]}
        relevant = {"q1": {"a"}, "q2": {"z"}, "q3": {"p"}}

        self.assertAlmostEqual(recall_at_k(ranked, relevant, 1), 1 / 3)
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 3), 2 / 3)

    def test_recall_counts_every_relevant_chunk(self):
        self.assertEqual(recall_at_k({"q": ["a", "x", "b"]}, {"q": {"a", "b"}}, 1), 0.5)
        self.assertEqual(recall_at_k({"q": ["a", "x", "b"]}, {"q": {"a", "b"}}, 3), 1.0)

    def test_every_query_needs_relevant_chunks(self):
        with self.assertRaises(ValueError):
            recall_at_k({"q": ["a"]}, {"q": set()}, 1)
        with self.assertRaises(ValueError):
            recall_at_k({}, {"q": {"a"}}, 1)


# --- Search service with a real SQLite database --------------------------------


class BasicSearchTests(SearchTestCase):
    def test_empty_database_returns_no_results_without_loading_the_model(self):
        results = self.run_search()

        self.assertEqual(results.results, ())
        self.assertEqual((results.query, results.top_k, results.document_id),
                         ("What is FastAPI?", DEFAULT_TOP_K, None))
        self.assertEqual(self.provider.model_inputs, [], "nothing to search: no query embedding")

    def test_chunks_without_embeddings_are_not_searched(self):
        self.one_chunk_document("A chunk that was never embedded.")

        self.assertEqual(self.run_search().results, ())

    def test_one_chunk(self):
        chunk = self.one_chunk_document("FastAPI is a Python web framework.")
        self.store_vector(chunk, VECTOR_B)

        [result] = self.run_search().results

        self.assertEqual((result.rank, result.chunk.chunk_id), (1, chunk.chunk_id))
        self.assertAlmostEqual(result.score, 0.6, places=6)
        self.assertEqual(result.chunk.text, chunk.text)
        self.assertEqual(result.chunk.source_locations, chunk.source_locations)

    def test_multiple_chunks_are_ranked_by_exact_similarity(self):
        chunks = self.corpus()

        results = self.run_search().results

        self.assertEqual(self.result_ids(results),
                         [chunks[name].chunk_id for name in "ABCD"])
        self.assertEqual([r.rank for r in results], [1, 2, 3, 4])
        expected_scores = [0.9 / math.sqrt(0.82), 0.6, 0.0, -1.0]
        for result, expected in zip(results, expected_scores):
            self.assertAlmostEqual(result.score, expected, places=5)
            self.assertIsInstance(result.score, float)

    def test_scores_are_the_dot_products_with_the_query_vector(self):
        chunks = self.corpus()
        vectors = {chunks[n].chunk_id: v for n, v in zip("ABCD", (VECTOR_A, VECTOR_B,
                                                                   VECTOR_C, VECTOR_D))}
        self.provider.query_vector = unit(0.3, -0.7, 0.2, 0.5)

        results = self.run_search().results

        for result in results:
            self.assertAlmostEqual(result.score,
                                   dot(self.provider.query_vector, vectors[result.chunk.chunk_id]),
                                   places=6)
        self.assertEqual([r.score for r in results], sorted((r.score for r in results),
                                                            reverse=True))

    def test_top_k_one(self):
        chunks = self.corpus()

        results = self.run_search(top_k=1).results

        self.assertEqual(self.result_ids(results), [chunks["A"].chunk_id])

    def test_top_k_five_with_more_chunks(self):
        chunks = self.corpus()
        extra = [self.one_chunk_document(f"Extra chunk {n}.") for n in range(3)]
        for chunk in extra:
            self.store_vector(chunk, unit(0.0, 1.0))  # score 0.0, same as C

        results = self.run_search(top_k=5).results

        ties = sorted([chunks["C"].chunk_id, *(c.chunk_id for c in extra)])
        self.assertEqual(self.result_ids(results),
                         [chunks["A"].chunk_id, chunks["B"].chunk_id, *ties[:3]])

    def test_top_k_larger_than_the_number_of_chunks_returns_all(self):
        chunks = self.corpus()

        results = self.run_search(top_k=50)

        self.assertEqual(results.top_k, 50)
        self.assertEqual(self.result_ids(results), [chunks[n].chunk_id for n in "ABCD"])

    def test_equal_scores_are_ranked_by_chunk_id(self):
        same = [self.one_chunk_document(f"Same vector {n}.") for n in range(4)]
        for chunk in reversed(same):  # stored in reverse order on purpose
            self.store_vector(chunk, VECTOR_B)

        results = self.run_search()

        self.assertEqual(self.result_ids(results), sorted(c.chunk_id for c in same))

    def test_same_search_twice_gives_identical_results(self):
        self.corpus()

        self.assertEqual(self.run_search(), self.run_search())

    def test_truncated_chunks_are_searched_and_flagged(self):
        chunk = self.one_chunk_document("A long chunk the model could not read completely.")
        self.store_vector(chunk, VECTOR_A, truncated=True)

        [result] = self.run_search().results

        self.assertTrue(result.truncated)


class QueryEmbeddingTests(SearchTestCase):
    def test_query_is_embedded_once_with_the_query_prefix(self):
        self.corpus()
        self.provider.model_inputs.clear()

        self.run_search("  What is FastAPI?  ")

        self.assertEqual(self.provider.model_inputs, ["query: What is FastAPI?"])

    def test_chinese_query_uses_the_query_prefix(self):
        self.corpus()
        self.provider.model_inputs.clear()

        self.run_search("FastAPI 是做什么的？")

        self.assertEqual(self.provider.model_inputs, ["query: FastAPI 是做什么的？"])

    def test_passage_prefix_is_never_used_for_the_query(self):
        self.corpus()
        self.provider.model_inputs.clear()
        self.provider.batch_calls.clear()

        self.run_search()

        self.assertEqual(self.provider.batch_calls, [], "embed_documents must not be called")
        self.assertFalse(any("passage:" in text for text in self.provider.model_inputs))

    def test_query_vector_comes_from_the_provider_contract(self):
        # A real (fake) provider: every score must be dot(query("..."), passage(text)).
        provider = FakeEmbeddingProvider()
        chunks = [self.one_chunk_document(f"Text number {n}.") for n in range(3)]
        for chunk in chunks:
            self.store_vector(chunk, provider.passage_vector(chunk.text), provider.contract)

        results = self.run_search("What is FastAPI?", provider=provider).results

        query_vector = provider.embed_query("What is FastAPI?")
        wrong_prefix = provider._vector("passage: What is FastAPI?")
        for result in results:
            passage = provider.passage_vector(result.chunk.text)
            self.assertAlmostEqual(result.score, dot(query_vector, passage), places=6)
            if abs(dot(query_vector, passage) - dot(wrong_prefix, passage)) > 1e-3:
                self.assertNotAlmostEqual(result.score, dot(wrong_prefix, passage), places=4)

    def test_same_query_gives_the_same_vector(self):
        provider = FakeEmbeddingProvider()

        self.assertEqual(provider.embed_query("What is FastAPI?"),
                         provider.embed_query("What is FastAPI?"))


class ReadOnlyTests(SearchTestCase):
    def test_search_changes_nothing_in_the_database(self):
        self.corpus()
        before = self.database_snapshot()
        file_before = hashlib.sha256(self.db_path.read_bytes()).hexdigest()

        self.run_search("unique-question-zebra-7731")

        self.assertEqual(self.database_snapshot(), before)
        self.assertEqual(hashlib.sha256(self.db_path.read_bytes()).hexdigest(), file_before)

    def test_query_text_is_not_stored(self):
        self.corpus()

        self.run_search("unique-question-zebra-7731")

        self.assertNotIn(b"zebra-7731", self.db_path.read_bytes())
        stored = [path for path in self.root.rglob("*") if path.is_file()
                  and b"zebra-7731" in path.read_bytes()]
        self.assertEqual(stored, [])

    def test_stale_embeddings_are_not_touched(self):
        chunks = self.corpus()
        self.execute("UPDATE embeddings SET model_revision = 'old' WHERE chunk_id = ?",
                     (chunks["A"].chunk_id,))
        before = self.database_snapshot()

        self.run_search()

        self.assertEqual(self.database_snapshot(), before, "search must not re-embed or delete")


class DocumentFilterTests(SearchTestCase):
    def setUp(self):
        super().setUp()
        self.first = self.one_chunk_document("First document.")
        self.second = self.one_chunk_document("Second document.")
        self.store_vector(self.first, VECTOR_B)
        self.store_vector(self.second, VECTOR_A)

    def test_without_filter_every_document_is_searched(self):
        results = self.run_search()

        self.assertEqual(self.result_ids(results), [self.second.chunk_id, self.first.chunk_id])

    def test_filter_searches_only_that_document(self):
        results = self.run_search(document_id=self.first.document_id)

        self.assertEqual(results.document_id, self.first.document_id)
        self.assertEqual(self.result_ids(results), [self.first.chunk_id])

    def test_document_without_valid_embeddings_gives_no_results(self):
        third = self.one_chunk_document("Third document, never embedded.")

        results = self.run_search(document_id=third.document_id)

        self.assertEqual(results.results, ())

    def test_unknown_document_is_not_found(self):
        with self.assertRaises(DocumentNotFoundError):
            self.run_search(document_id="0" * 32)
        self.assertEqual(self.provider.model_inputs, [])

    def test_malformed_or_injected_document_id_is_rejected(self):
        for bad in ["not-an-id", "' OR '1'='1", f"{self.first.document_id}' OR 1=1 --",
                    self.first.document_id.upper(), ""]:
            with self.subTest(document_id=bad), self.assertRaises(InvalidDocumentIdError):
                self.run_search(document_id=bad)

    def test_document_id_is_data_not_sql_even_without_validation(self):
        # storage.load_candidates is called directly, skipping the ID check:
        # the "?" placeholder still treats the text as a plain value.
        contract = self.provider.contract
        for injected in ["' OR '1'='1", "x' UNION SELECT * FROM chunks --"]:
            with self.subTest(document_id=injected):
                self.assertEqual(storage.load_candidates(self.db_path, contract, injected),
                                 ([], 0))
        self.assertEqual(len(storage.load_candidates(self.db_path, contract)[0]), 2)


class ContractSafetyTests(SearchTestCase):
    """A vector that is not valid for the current contract never takes part."""

    def setUp(self):
        super().setUp()
        self.chunks_by_name = self.corpus()
        self.target = self.chunks_by_name["A"]  # the chunk that would rank first

    def assert_target_excluded(self):
        ids = self.result_ids(self.run_search())
        self.assertNotIn(self.target.chunk_id, ids)
        self.assertEqual(ids, [self.chunks_by_name[n].chunk_id for n in "BCD"],
                         "every other valid chunk is still found, in order")

    def update_target(self, assignment, parameters=(), ignore_checks=False):
        self.execute(f"UPDATE embeddings SET {assignment} WHERE chunk_id = ?",
                     (*parameters, self.target.chunk_id), ignore_checks=ignore_checks)

    def test_valid_target_is_found_first(self):
        self.assertEqual(self.result_ids(self.run_search())[0], self.target.chunk_id)

    def test_contract_mismatches_are_excluded(self):
        valid_input_sha = text_sha256("passage: " + self.target.text)
        mismatches = {
            "stale model (another model_name)": ("model_name = ?", ("other/model",)),
            "wrong model revision": ("model_revision = ?", ("0" * 40,)),
            "Batch 5 row without revision": ("model_revision = ''", ()),
            "wrong embedding version": ("embedding_version = ?", (1,)),
            "wrong max_tokens": ("max_tokens = ?", (128,)),
            "wrong normalized flag": ("normalized = 0", ()),
            "wrong passage prefix": ("passage_prefix = ?", ("",)),
            "query prefix stored as passage prefix": ("passage_prefix = ?", ("query: ",)),
            "text hash mismatch": ("text_sha256 = ?", (text_sha256("other text"),)),
            "input hash mismatch": ("input_sha256 = ?", (text_sha256(self.target.text),)),
        }
        for label, (assignment, parameters) in mismatches.items():
            with self.subTest(label):
                self.update_target(assignment, parameters)
                self.assert_target_excluded()
                # restore the valid row for the next case
                self.store_vector(self.target, VECTOR_A)
                self.assertEqual(embedding_storage.get_embedding(
                    self.db_path, self.target.chunk_id).record.input_sha256, valid_input_sha)

    def test_wrong_dimension_is_excluded(self):
        # A consistent 4-number row (the table's CHECK allows it) of another contract.
        blob = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        self.update_target("dimension = 4, vector = ?", (blob,))

        self.assert_target_excluded()

    def test_changed_chunk_text_makes_the_vector_stale(self):
        new_text = "Completely different text now."
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("UPDATE chunks SET text = ?, char_count = ? WHERE chunk_id = ?",
                               (new_text, len(new_text), self.target.chunk_id))

        self.assert_target_excluded()

    def test_stale_vector_with_a_perfect_score_is_still_excluded(self):
        other = self.one_chunk_document("Stored by an older model.")
        self.store_vector(other, QUERY_VECTOR)  # would score exactly 1.0
        self.execute("UPDATE embeddings SET embedding_version = 1 WHERE chunk_id = ?",
                     (other.chunk_id,))

        self.assertNotIn(other.chunk_id, self.result_ids(self.run_search()))

    def test_stale_row_with_corrupted_bytes_is_skipped_not_read(self):
        nan_blob = struct.pack("<8f", float("nan"), *[0.0] * 7)
        self.update_target("model_revision = 'old', vector = ?", (nan_blob,))

        self.assert_target_excluded()

    def test_non_normalized_contract_cannot_be_searched(self):
        provider = ScriptedProvider(normalize=False)

        with self.assertRaises(SearchNotSupportedError):
            self.run_search(provider=provider)
        self.assertEqual(provider.model_inputs, [])


class NumericalSafetyTests(SearchTestCase):
    """Corrupted vectors that pass the contract check stop the search safely."""

    def setUp(self):
        super().setUp()
        self.chunks_by_name = self.corpus()
        self.target = self.chunks_by_name["A"]

    def corrupt_target(self, blob, ignore_checks=False):
        self.execute("UPDATE embeddings SET vector = ? WHERE chunk_id = ?",
                     (blob, self.target.chunk_id), ignore_checks=ignore_checks)

    def assert_safe_storage_error(self):
        with self.assertRaises(StorageError) as caught:
            self.run_search()
        self.assertEqual(caught.exception.safe_message,
                         embedding_storage.CORRUPTED_EMBEDDINGS_MESSAGE)

    def test_wrong_byte_length(self):
        for blob in [b"", b"\x00" * 31, b"\x00" * 33, struct.pack("<16f", *[0.25] * 16)]:
            with self.subTest(size=len(blob)):
                self.corrupt_target(blob, ignore_checks=True)
                self.assert_safe_storage_error()

    def test_wrong_byte_order(self):
        self.corrupt_target(struct.pack(">8f", *VECTOR_A))  # big-endian: garbage numbers

        self.assert_safe_storage_error()

    def test_nan_is_rejected(self):
        self.corrupt_target(struct.pack("<8f", float("nan"), *[0.0] * 7))

        self.assert_safe_storage_error()

    def test_infinity_is_rejected(self):
        for value in [float("inf"), float("-inf")]:
            with self.subTest(value=value):
                self.corrupt_target(struct.pack("<8f", value, *[0.0] * 7))
                self.assert_safe_storage_error()

    def test_non_unit_stored_vector_is_rejected_not_renormalized(self):
        for vector in [(2.0,) + (0.0,) * 7, (0.0,) * 8, (0.9, 0.1) + (0.0,) * 6]:
            with self.subTest(vector=vector):
                self.corrupt_target(struct.pack("<8f", *vector))
                self.assert_safe_storage_error()

    def test_vector_stored_as_text_is_rejected(self):
        self.corrupt_target("x" * 32, ignore_checks=True)

        self.assert_safe_storage_error()

    def test_corrupted_chunk_row_is_rejected(self):
        self.execute("UPDATE chunks SET source_locations = 'not json' WHERE chunk_id = ?",
                     (self.target.chunk_id,))

        self.assert_safe_storage_error()

    def test_invalid_query_vectors_are_rejected(self):
        bad_vectors = {
            "wrong dimension": unit(1.0, dimension=4),
            "too many numbers": unit(1.0, dimension=9),
            "NaN": (float("nan"),) + (0.0,) * 7,
            "infinity": (float("inf"),) + (0.0,) * 7,
            "not normalized": (2.0,) + (0.0,) * 7,
            "zero vector": (0.0,) * 8,
        }
        for label, vector in bad_vectors.items():
            with self.subTest(label), self.assertRaises(InvalidQueryVectorError):
                self.run_search(provider=ScriptedProvider(query_vector=vector))

    def test_unexpected_provider_error_becomes_a_safe_error(self):
        class BrokenProvider(ScriptedProvider):
            def embed_query(self, text):
                raise RuntimeError("secret internal detail C:\\private\\path")

        with self.assertRaises(ProcessingError) as caught:
            self.run_search(provider=BrokenProvider())

        self.assertEqual(caught.exception.safe_message, service.UNEXPECTED_SEARCH_ERROR_MESSAGE)
        self.assertNotIn("secret", caught.exception.safe_message)

    def test_database_failure_is_reported_safely(self):
        self.db_path.unlink()
        self.db_path.mkdir()  # a folder where the database file should be

        with self.assertRaises(StorageError) as caught:
            self.run_search()

        self.assertNotIn(str(self.root), caught.exception.safe_message)


class CandidateLoadingTests(SearchTestCase):
    def test_candidates_carry_chunk_data_but_come_from_one_query(self):
        chunks = self.corpus()
        statements = []
        original_connect = database.connect

        def counting_connect(path):
            context = original_connect(path)

            class Counting:
                def __enter__(self):
                    connection = context.__enter__()
                    connection.set_trace_callback(statements.append)
                    return connection

                def __exit__(self, *exc):
                    return context.__exit__(*exc)

            return Counting()

        from unittest.mock import patch
        with patch.object(storage, "connect", counting_connect):
            candidates, stale = storage.load_candidates(self.db_path, self.provider.contract)

        selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
        self.assertEqual(len(selects), 1, "one SELECT for all candidates (no N+1)")
        self.assertEqual(stale, 0)
        self.assertEqual(sorted(c.chunk.chunk_id for c in candidates),
                         sorted(c.chunk_id for c in chunks.values()))
        by_id = {c.chunk.chunk_id: c for c in candidates}
        self.assertEqual(by_id[chunks["A"].chunk_id].vector, VECTOR_A)
        self.assertEqual(by_id[chunks["A"].chunk_id].chunk, chunks["A"])

    def test_stale_rows_are_counted(self):
        chunks = self.corpus()
        self.execute("UPDATE embeddings SET embedding_version = 1 WHERE chunk_id IN (?, ?)",
                     (chunks["A"].chunk_id, chunks["B"].chunk_id))

        candidates, stale = storage.load_candidates(self.db_path, self.provider.contract)

        self.assertEqual((len(candidates), stale), (2, 2))


if __name__ == "__main__":
    unittest.main()
