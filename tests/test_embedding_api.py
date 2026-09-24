"""Tests for the embedding endpoints in app/api.py.

    POST /documents/{document_id}/embed
    GET  /documents/{document_id}/embeddings

The real model is replaced with FakeEmbeddingProvider through FastAPI's
dependency_overrides, so no model is downloaded or loaded.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import unittest
from unittest.mock import patch

from app import api
from app.embeddings import service
from app.embeddings.config import MAX_BATCH_SIZE
from app.embeddings.models import EmbeddingModelUnavailableError
from fake_embeddings import FAKE_MODEL_NAME, FakeEmbeddingProvider
from test_chunking_api import ChunkApiTestCase

CONTRACT_FIELDS = {"document_id", "status", "model_name", "embedding_version", "dimension",
                   "dtype", "normalized", "total_chunks"}


class EmbeddingApiTestCase(ChunkApiTestCase):
    def setUp(self):
        super().setUp()
        self.provider = FakeEmbeddingProvider()
        api.app.dependency_overrides[api.get_embedding_provider] = lambda: self.provider
        self.addCleanup(api.app.dependency_overrides.clear)

    def chunked_document(self, **chunk_params):
        document_id = self.processed_document()
        self.assertEqual(self.chunk(document_id, **(chunk_params or {"chunk_size": 300,
                                                                      "chunk_overlap": 0})
                                    ).status_code, 200)
        return document_id

    def embed(self, document_id, **params):
        return self.client.post(f"/documents/{document_id}/embed", params=params)

    def embedding_status(self, document_id):
        return self.client.get(f"/documents/{document_id}/embeddings")


class EmbedEndpointTests(EmbeddingApiTestCase):
    def test_chunked_document_is_embedded(self):
        document_id = self.chunked_document()
        total = self.list_chunks(document_id).json()["chunk_count"]

        response = self.embed(document_id)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), CONTRACT_FIELDS | {"embedded_count", "skipped_count",
                                                       "stale_reembedded_count"})
        self.assertEqual(body["status"], "complete")
        self.assertEqual(body["model_name"], FAKE_MODEL_NAME)
        self.assertEqual((body["dimension"], body["dtype"], body["normalized"]),
                         (8, "float32", True))
        self.assertEqual((body["total_chunks"], body["embedded_count"], body["skipped_count"],
                          body["stale_reembedded_count"]), (total, total, 0, 0))

    def test_embedding_again_skips_everything(self):
        document_id = self.chunked_document()
        self.embed(document_id)

        body = self.embed(document_id).json()

        self.assertEqual((body["embedded_count"], body["skipped_count"]),
                         (0, body["total_chunks"]))

    def test_response_never_contains_vectors_text_or_paths(self):
        document_id = self.chunked_document()

        response = self.embed(document_id)

        self.assertNotIn("vector", response.text)
        self.assertNotIn("word42", response.text)
        self.assertNotIn(str(self.temp_root), response.text)
        self.assertNotIn("cache", response.text)

    def test_batch_size_is_used(self):
        document_id = self.chunked_document()

        self.embed(document_id, batch_size=2)

        self.assertEqual(max(self.provider.batch_calls), 2)

    def test_invalid_batch_size_returns_422(self):
        document_id = self.chunked_document()
        message = f"batch_size must be a whole number from 1 to {MAX_BATCH_SIZE}."
        for bad in [0, -1, MAX_BATCH_SIZE + 1]:
            with self.subTest(batch_size=bad):
                self.assert_safe_error(self.embed(document_id, batch_size=bad), 422, message)
        self.assertEqual(self.embed(document_id, batch_size="many").status_code, 422)
        self.assertEqual(self.provider.batch_calls, [])

    def test_missing_document_returns_404(self):
        self.assert_safe_error(self.embed("0" * 32), 404, "Document not found.")

    def test_invalid_document_id_returns_400(self):
        self.assert_safe_error(self.embed("not-an-id"), 400, "Invalid document ID.")

    def test_unprocessed_document_returns_409(self):
        document_id = self.upload_document("notes.txt", b"Hello")

        self.assert_safe_error(self.embed(document_id), 409,
                               "This document has not been processed yet.")

    def test_document_without_chunks_returns_409(self):
        document_id = self.processed_document()

        self.assert_safe_error(self.embed(document_id), 409,
                               "This document has no chunks yet. Chunk it first.")

    def test_model_that_cannot_load_returns_503(self):
        document_id = self.chunked_document()

        with patch.object(self.provider, "embed_documents",
                          side_effect=EmbeddingModelUnavailableError()):
            response = self.embed(document_id)

        self.assert_safe_error(response, 503, EmbeddingModelUnavailableError.safe_message)

    def test_unexpected_model_error_returns_generic_500(self):
        document_id = self.chunked_document()

        with patch.object(self.provider, "embed_documents",
                          side_effect=RuntimeError("secret C:\\internal\\path")):
            response = self.embed(document_id)

        self.assert_safe_error(response, 500, service.UNEXPECTED_EMBEDDING_ERROR_MESSAGE)
        self.assertNotIn("secret", response.text)


class EmbeddingStatusEndpointTests(EmbeddingApiTestCase):
    def test_status_before_and_after_embedding(self):
        document_id = self.chunked_document()
        total = self.list_chunks(document_id).json()["chunk_count"]

        before = self.embedding_status(document_id).json()
        self.embed(document_id)
        after = self.embedding_status(document_id).json()

        self.assertEqual(set(after), CONTRACT_FIELDS | {"embedded_chunks", "missing_embeddings",
                                                        "stale_embeddings", "truncated_chunks"})
        self.assertEqual((before["status"], before["missing_embeddings"]), ("incomplete", total))
        self.assertEqual((after["status"], after["embedded_chunks"], after["missing_embeddings"],
                          after["stale_embeddings"]), ("complete", total, 0, 0))
        self.assertNotIn("vector", self.embedding_status(document_id).text)

    def test_status_shows_stale_vectors_after_model_change(self):
        document_id = self.chunked_document()
        self.embed(document_id)
        total = self.list_chunks(document_id).json()["chunk_count"]

        self.provider = FakeEmbeddingProvider(model_name="fake/newer-model")
        body = self.embedding_status(document_id).json()

        self.assertEqual((body["status"], body["stale_embeddings"], body["model_name"]),
                         ("incomplete", total, "fake/newer-model"))
        self.assertEqual(self.embed(document_id).json()["stale_reembedded_count"], total)

    def test_status_of_unchunked_document_is_no_chunks(self):
        document_id = self.processed_document()

        body = self.embedding_status(document_id).json()

        self.assertEqual((body["status"], body["total_chunks"]), ("no_chunks", 0))

    def test_status_errors_are_safe(self):
        self.assert_safe_error(self.embedding_status("0" * 32), 404, "Document not found.")
        self.assert_safe_error(self.embedding_status("bad-id"), 400, "Invalid document ID.")


class RealProviderWiringTests(unittest.TestCase):
    def test_default_provider_uses_the_git_ignored_model_cache(self):
        api.get_embedding_provider.cache_clear()
        self.addCleanup(api.get_embedding_provider.cache_clear)

        provider = api.get_embedding_provider()

        self.assertIs(api.get_embedding_provider(), provider, "one provider per process")
        self.assertEqual(provider.config.cache_dir, api.STORAGE_DIR / "model_cache")
        self.assertEqual(provider.contract.dimension, 384)


if __name__ == "__main__":
    unittest.main()
