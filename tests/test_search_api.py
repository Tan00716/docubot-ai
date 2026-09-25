"""Tests for POST /search in app/api.py.

The real model is replaced with FakeEmbeddingProvider through FastAPI's
dependency_overrides, so no model is downloaded or loaded. Documents go
through the real upload -> process -> chunk -> embed endpoints first.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import math
import struct
import unittest
from contextlib import closing
import sqlite3

from app import api
from app.embeddings.models import EmbeddingModelUnavailableError
from app.search.models import MAX_QUERY_LENGTH, MAX_TOP_K
from fake_embeddings import FakeEmbeddingProvider
from test_embedding_api import EmbeddingApiTestCase

RESPONSE_FIELDS = {"query", "top_k", "document_id", "results"}
RESULT_FIELDS = {"rank", "score", "chunk_id", "document_id", "chunk_index", "source_filename",
                 "source_locations", "truncated", "text"}
QUERY_MESSAGE = f"query must be a non-empty string of at most {MAX_QUERY_LENGTH} characters."


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


class SearchApiTestCase(EmbeddingApiTestCase):
    def embedded_document(self):
        document_id = self.chunked_document()
        self.assertEqual(self.embed(document_id).status_code, 200)
        return document_id

    def search(self, **body):
        return self.client.post("/search", json=body)

    def chunk_texts(self, document_id):
        chunks = self.list_chunks(document_id, include_text=True).json()["chunks"]
        return {chunk["chunk_id"]: chunk for chunk in chunks}

    def assert_validation_error(self, response):
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn(str(self.temp_root), response.text)
        self.assertEqual(self.provider.model_inputs, [], "invalid requests never reach the model")


class SearchEndpointTests(SearchApiTestCase):
    def test_search_returns_ranked_chunks(self):
        document_id = self.embedded_document()
        chunks = self.chunk_texts(document_id)
        self.provider.model_inputs.clear()

        response = self.search(query="What is FastAPI?", top_k=3)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), RESPONSE_FIELDS)
        self.assertEqual((body["query"], body["top_k"], body["document_id"]),
                         ("What is FastAPI?", 3, None))
        self.assertEqual(len(body["results"]), 3)
        self.assertEqual(self.provider.model_inputs, ["query: What is FastAPI?"])

        query_vector = self.provider.embed_query("What is FastAPI?")
        expected = sorted(((round(dot(query_vector, self.provider.passage_vector(c["text"])), 6),
                            chunk_id) for chunk_id, c in chunks.items()),
                          key=lambda pair: (-pair[0], pair[1]))[:3]
        self.assertEqual([(r["score"], r["chunk_id"]) for r in body["results"]], expected)
        for rank, result in enumerate(body["results"], start=1):
            chunk = chunks[result["chunk_id"]]
            self.assertEqual(set(result), RESULT_FIELDS)
            self.assertEqual(result["rank"], rank)
            self.assertIsInstance(result["score"], float)
            self.assertEqual(result["document_id"], document_id)
            self.assertEqual((result["chunk_index"], result["text"], result["source_locations"]),
                             (chunk["chunk_index"], chunk["text"], chunk["source_locations"]))
            self.assertEqual(result["source_filename"], "notes.txt")
            self.assertIs(result["truncated"], False)

    def test_default_top_k_is_five(self):
        self.embedded_document()

        body = self.search(query="What is FastAPI?").json()

        self.assertEqual((body["top_k"], len(body["results"])), (5, 5))

    def test_query_is_trimmed(self):
        self.embedded_document()

        body = self.search(query="   What is FastAPI?\n").json()

        self.assertEqual(body["query"], "What is FastAPI?")

    def test_chinese_query(self):
        self.embedded_document()
        self.provider.model_inputs.clear()

        response = self.search(query="FastAPI 是做什么的？", top_k=1)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["query"], "FastAPI 是做什么的？")
        self.assertEqual(self.provider.model_inputs, ["query: FastAPI 是做什么的？"])

    def test_empty_database_returns_200_with_no_results(self):
        response = self.search(query="What is FastAPI?")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(),
                         {"query": "What is FastAPI?", "top_k": 5, "document_id": None,
                          "results": []})

    def test_document_filter(self):
        first = self.embedded_document()
        second = self.embedded_document()

        filtered = self.search(query="What is FastAPI?", top_k=50, document_id=first).json()
        everything = self.search(query="What is FastAPI?", top_k=50).json()

        self.assertEqual(filtered["document_id"], first)
        self.assertEqual({r["document_id"] for r in filtered["results"]}, {first})
        self.assertEqual({r["document_id"] for r in everything["results"]}, {first, second})
        self.assertEqual(len(everything["results"]),
                         len(filtered["results"]) + len(self.chunk_texts(second)))

    def test_document_filter_without_embeddings_returns_no_results(self):
        document_id = self.chunked_document()

        body = self.search(query="What is FastAPI?", document_id=document_id).json()

        self.assertEqual(body["results"], [])

    def test_unknown_document_returns_404(self):
        self.embedded_document()

        self.assert_safe_error(self.search(query="q", document_id="0" * 32), 404,
                               "Document not found.")

    def test_malformed_document_id_returns_400(self):
        for bad in ["not-an-id", "' OR '1'='1", "1; DROP TABLE chunks; --", "A" * 32, ""]:
            with self.subTest(document_id=bad):
                self.assert_safe_error(self.search(query="q", document_id=bad), 400,
                                       "Invalid document ID.")

    def test_sql_injection_attempt_leaves_the_database_intact(self):
        document_id = self.embedded_document()
        before = len(self.chunk_texts(document_id))

        self.search(query="x' OR '1'='1", document_id="x'; DELETE FROM chunks; --")
        self.search(query="'; DROP TABLE embeddings; --")

        self.assertEqual(len(self.chunk_texts(document_id)), before)
        self.assertEqual(self.search(query="q", top_k=50).status_code, 200)

    def test_search_is_repeatable(self):
        self.embedded_document()

        first = self.search(query="What is FastAPI?", top_k=10).json()
        second = self.search(query="What is FastAPI?", top_k=10).json()

        self.assertEqual(first, second)


class SearchValidationTests(SearchApiTestCase):
    def setUp(self):
        super().setUp()
        self.embedded_document()
        self.provider.model_inputs.clear()

    def test_empty_or_blank_query_returns_422(self):
        for bad in ["", " ", "\n\t "]:
            with self.subTest(query=repr(bad)):
                response = self.search(query=bad)
                self.assert_validation_error(response)

        self.assertIn(QUERY_MESSAGE, self.search(query="   ").text)

    def test_too_long_query_returns_422(self):
        self.assert_validation_error(self.search(query="x" * (MAX_QUERY_LENGTH + 1)))
        self.assertEqual(self.search(query="x" * MAX_QUERY_LENGTH).status_code, 200)

    def test_invalid_top_k_returns_422(self):
        for bad in [0, -1, -100, MAX_TOP_K + 1, 1_000_000, "5", 5.5, 5.0, True, None]:
            with self.subTest(top_k=bad):
                self.assert_validation_error(self.search(query="q", top_k=bad))

    def test_maximum_top_k_is_accepted(self):
        self.assertEqual(self.search(query="q", top_k=MAX_TOP_K).status_code, 200)

    def test_wrong_types_and_missing_fields_return_422(self):
        for body in [{}, {"query": None}, {"query": 42}, {"query": ["a"]},
                     {"query": "q", "document_id": 7}, {"query": "q", "topk": 5},
                     {"query": "q", "vector": [0.1] * 8}]:
            with self.subTest(body=body):
                self.assert_validation_error(self.client.post("/search", json=body))

    def test_malformed_json_returns_422(self):
        for raw in [b"{not json", b"", b"[1, 2]", b'"just a string"', b"\xff\xfe"]:
            with self.subTest(raw=raw):
                response = self.client.post("/search", content=raw,
                                            headers={"Content-Type": "application/json"})
                self.assert_validation_error(response)

    def test_oversized_request_is_rejected_before_the_model(self):
        response = self.search(query="x" * 200_000, top_k=10**12)

        self.assert_validation_error(response)


class SearchSafetyTests(SearchApiTestCase):
    def test_response_never_contains_vectors_or_paths(self):
        self.embedded_document()

        response = self.search(query="What is FastAPI?", top_k=50)

        body = response.json()
        for result in body["results"]:
            self.assertFalse({"vector", "embedding", "embeddings"} & set(result))
            self.assertTrue(all(not isinstance(v, list) or k == "source_locations"
                                for k, v in result.items()))
        self.assertNotIn("vector", response.text)
        self.assertNotIn(str(self.temp_root), response.text)
        self.assertNotIn("storage", response.text)
        self.assertNotIn("cache", response.text)
        self.assertNotIn(".db", response.text)

    def test_corrupted_vector_returns_a_safe_500(self):
        document_id = self.embedded_document()
        chunk_id = next(iter(self.chunk_texts(document_id)))
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("UPDATE embeddings SET vector = ? WHERE chunk_id = ?",
                               (struct.pack("<8f", float("nan"), *[0.0] * 7), chunk_id))

        self.assert_safe_error(self.search(query="q"), 500, "Stored embedding data is corrupted.")

    def test_unexpected_error_hides_internal_details(self):
        self.embedded_document()

        def broken(text):
            raise RuntimeError(f"secret token 123:ABC in {self.temp_root}")

        self.provider.embed_query = broken

        response = self.search(query="q")

        self.assert_safe_error(response, 500, "Unexpected error while searching.")
        self.assertNotIn("secret", response.text)
        self.assertNotIn("RuntimeError", response.text)

    def test_unavailable_model_returns_503(self):
        self.embedded_document()

        def unavailable(text):
            raise EmbeddingModelUnavailableError()

        self.provider.embed_query = unavailable

        self.assert_safe_error(self.search(query="q"), 503,
                               EmbeddingModelUnavailableError.safe_message)

    def test_invalid_query_vector_returns_a_safe_500(self):
        self.embedded_document()
        self.provider.embed_query = lambda text: (float("inf"),) + (0.0,) * 7

        self.assert_safe_error(self.search(query="q"), 500,
                               "The embedding model returned an invalid query vector.")

    def test_non_normalized_contract_returns_a_safe_500(self):
        self.provider = FakeEmbeddingProvider(normalize=False)
        api.app.dependency_overrides[api.get_embedding_provider] = lambda: self.provider

        self.assert_safe_error(self.search(query="q"), 500,
                               "Vector search requires normalized embeddings.")


class SearchOpenApiTests(SearchApiTestCase):
    def test_search_is_documented_with_its_limits(self):
        schema = self.client.get("/openapi.json").json()

        operation = schema["paths"]["/search"]["post"]
        self.assertIn("requestBody", operation)
        request = schema["components"]["schemas"]["SearchRequest"]
        self.assertEqual(request["required"], ["query"])
        self.assertIs(request["additionalProperties"], False)
        self.assertEqual((request["properties"]["query"]["minLength"],
                          request["properties"]["query"]["maxLength"]), (1, MAX_QUERY_LENGTH))
        top_k = request["properties"]["top_k"]
        self.assertEqual((top_k["minimum"], top_k["maximum"], top_k["default"]),
                         (1, MAX_TOP_K, 5))
        self.assertIn("document_id", request["properties"])
        result = schema["components"]["schemas"]["SearchResultResponse"]
        self.assertEqual(set(result["properties"]), RESULT_FIELDS)
        self.assertEqual(self.client.get("/docs").status_code, 200)


if __name__ == "__main__":
    unittest.main()
