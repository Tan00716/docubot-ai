"""Tests for the chunking endpoints in app/api.py.

    POST /documents/{document_id}/chunk
    GET  /documents/{document_id}/chunks

Every test uses temporary folders and a temporary SQLite database (see
ApiTestCase). No Telegram, no .env, no network.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import unittest
from unittest.mock import patch

from app import api
from app.document_processing import processor
from app.document_processing.chunking import CHUNKING_VERSION
from sample_documents import make_pdf
from test_api import DocumentApiTestCase

LONG_TEXT = " ".join(f"word{n}" for n in range(600)).encode()  # about 4,500 characters

SUMMARY_FIELDS = {
    "document_id", "chunking_status", "chunk_count", "chunking_version",
    "chunk_size", "chunk_overlap", "chunked_at",
}


class ChunkApiTestCase(DocumentApiTestCase):
    def processed_document(self, filename="notes.txt", content=LONG_TEXT):
        document_id = self.upload_document(filename, content)
        self.assertEqual(self.process(document_id).status_code, 200)
        return document_id

    def chunk(self, document_id, **params):
        return self.client.post(f"/documents/{document_id}/chunk", params=params)

    def list_chunks(self, document_id, **params):
        return self.client.get(f"/documents/{document_id}/chunks", params=params)

    def assert_safe_error(self, response, status_code, message):
        self.assertEqual(response.status_code, status_code)
        self.assertEqual(response.json(), {"detail": message})
        self.assertNotIn(str(self.temp_root), response.text)
        self.assertNotIn("Traceback", response.text)


class ChunkEndpointSuccessTests(ChunkApiTestCase):
    def test_processed_document_is_chunked_with_default_settings(self):
        document_id = self.processed_document()

        response = self.chunk(document_id)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(set(body), SUMMARY_FIELDS)
        self.assertEqual(body["document_id"], document_id)
        self.assertEqual(body["chunking_status"], "chunked")
        self.assertGreater(body["chunk_count"], 1)
        self.assertEqual(body["chunking_version"], CHUNKING_VERSION)
        self.assertEqual((body["chunk_size"], body["chunk_overlap"]), (1200, 200))
        self.assertIsInstance(body["chunked_at"], str)

    def test_summary_never_contains_chunk_text(self):
        document_id = self.processed_document()

        response = self.chunk(document_id)

        self.assertNotIn("word42", response.text)
        self.assertNotIn(str(self.temp_root), response.text)

    def test_custom_settings_are_used(self):
        document_id = self.processed_document()

        body = self.chunk(document_id, chunk_size=300, chunk_overlap=50).json()

        self.assertEqual((body["chunk_size"], body["chunk_overlap"]), (300, 50))
        self.assertGreater(body["chunk_count"], 10)

    def test_chunking_twice_does_not_duplicate_chunks(self):
        document_id = self.processed_document()
        first = self.chunk(document_id).json()
        first_ids = [c["chunk_id"] for c in self.list_chunks(document_id).json()["chunks"]]

        second = self.chunk(document_id).json()
        second_ids = [c["chunk_id"] for c in self.list_chunks(document_id).json()["chunks"]]

        self.assertEqual(first["chunk_count"], second["chunk_count"])
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(len(second_ids), second["chunk_count"])

    def test_pdf_chunks_list_real_page_numbers(self):
        document_id = self.processed_document("guide.pdf", make_pdf(["One.", None, "Three."]))
        self.chunk(document_id)

        chunk = self.list_chunks(document_id).json()["chunks"][0]

        self.assertEqual([loc["page"] for loc in chunk["source_locations"]], [1, 3])


class ChunkListTests(ChunkApiTestCase):
    def test_listing_returns_metadata_without_text_by_default(self):
        document_id = self.processed_document()
        self.chunk(document_id)

        body = self.list_chunks(document_id).json()

        self.assertEqual(body["chunking_status"], "chunked")
        self.assertEqual(body["source_filename"], "notes.txt")
        self.assertEqual(body["file_type"], ".txt")
        self.assertEqual(len(body["chunks"]), body["chunk_count"])
        for index, chunk in enumerate(body["chunks"]):
            self.assertEqual(chunk["chunk_index"], index)
            self.assertEqual(
                chunk["chunk_id"], f"{document_id}_v{CHUNKING_VERSION}_s1200_o200_{index:05d}"
            )
            self.assertGreater(chunk["char_count"], 0)
            self.assertEqual(set(chunk["source_locations"][0]),
                             {"block", "line_start", "line_end", "char_start", "char_end"})
            self.assertIsNone(chunk["text"])
        self.assertNotIn("word42", self.list_chunks(document_id).text)

    def test_listing_can_include_text(self):
        document_id = self.processed_document("short.txt", b"Hello DocuBot.\n\nSecond block.")
        self.chunk(document_id)

        chunks = self.list_chunks(document_id, include_text="true").json()["chunks"]

        self.assertEqual([c["text"] for c in chunks], ["Hello DocuBot.\n\nSecond block."])

    def test_document_without_chunks_is_not_chunked(self):
        document_id = self.processed_document()

        response = self.list_chunks(document_id)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["chunking_status"], "not_chunked")
        self.assertEqual(body["chunk_count"], 0)
        self.assertEqual(body["chunks"], [])
        self.assertIsNone(body["chunk_size"])
        self.assertIsNone(body["chunked_at"])

    def test_processing_again_clears_the_chunks(self):
        document_id = self.processed_document()
        self.chunk(document_id)

        self.process(document_id)

        self.assertEqual(self.list_chunks(document_id).json()["chunking_status"], "not_chunked")

    def test_listing_unknown_or_invalid_document(self):
        self.assert_safe_error(self.list_chunks("0" * 32), 404, "Document not found.")
        self.assert_safe_error(self.list_chunks("not-an-id"), 400, "Invalid document ID.")

    def test_listing_database_failure_returns_safe_500(self):
        with patch.object(api, "DATABASE_PATH", self.temp_root):
            response = self.list_chunks("0" * 32)

        self.assert_safe_error(response, 500, "Could not access the document database.")


class ChunkEndpointFailureTests(ChunkApiTestCase):
    def test_unknown_document_returns_404(self):
        self.assert_safe_error(self.chunk("0" * 32), 404, "Document not found.")

    def test_invalid_document_id_returns_400(self):
        for bad_id in ["not-a-valid-id", "A" * 32, "%2e%2e"]:
            with self.subTest(document_id=bad_id):
                self.assert_safe_error(self.chunk(bad_id), 400, "Invalid document ID.")

    def test_unprocessed_document_returns_409(self):
        document_id = self.upload_document("notes.txt", LONG_TEXT)

        response = self.chunk(document_id)

        self.assert_safe_error(response, 409, "This document has not been processed yet.")

    def test_failed_document_returns_409(self):
        document_id = self.upload_document("broken.pdf", b"%PDF-1.4 broken")
        self.process(document_id)

        response = self.chunk(document_id)

        self.assert_safe_error(response, 409, "This document has not been processed yet.")

    def test_invalid_settings_return_422_and_store_nothing(self):
        document_id = self.processed_document()
        cases = [
            ({"chunk_size": 0}, "chunk_size must be a positive whole number."),
            ({"chunk_size": -10}, "chunk_size must be a positive whole number."),
            ({"chunk_size": 10_001}, "chunk_size must be at most 10000 characters."),
            ({"chunk_overlap": -1}, "chunk_overlap must be zero or a positive whole number."),
            ({"chunk_size": 100, "chunk_overlap": 100},
             "chunk_overlap must be at most half of chunk_size."),
            ({"chunk_overlap": 601}, "chunk_overlap must be at most half of chunk_size."),
            ({"chunk_size": 10_000, "chunk_overlap": 9_999},
             "chunk_overlap must be at most half of chunk_size."),
        ]
        for params, message in cases:
            with self.subTest(params=params):
                self.assert_safe_error(self.chunk(document_id, **params), 422, message)

        self.assertEqual(self.list_chunks(document_id).json()["chunk_count"], 0)

    def test_non_numeric_settings_are_rejected_by_fastapi(self):
        document_id = self.processed_document()

        response = self.chunk(document_id, chunk_size="big")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.list_chunks(document_id).json()["chunk_count"], 0)

    def test_invalid_settings_are_checked_before_the_document(self):
        response = self.chunk("0" * 32, chunk_size=0)

        self.assertEqual(response.status_code, 422)

    def test_missing_processed_file_returns_404(self):
        document_id = self.processed_document()
        (self.processed_dir / f"{document_id}.json").unlink()

        response = self.chunk(document_id)

        self.assert_safe_error(
            response, 404,
            "The processed output for this document is missing. Process the document again.",
        )

    def test_corrupted_processed_file_returns_safe_500(self):
        document_id = self.processed_document()
        (self.processed_dir / f"{document_id}.json").write_text("{ broken", encoding="utf-8")

        response = self.chunk(document_id)

        self.assert_safe_error(response, 500, processor.CORRUPTED_PROCESSED_MESSAGE)

    def test_unexpected_error_returns_generic_500(self):
        document_id = self.processed_document()
        crash = RuntimeError("secret internal info")

        with patch.object(processor, "build_chunks", side_effect=crash):
            response = self.chunk(document_id)

        self.assert_safe_error(response, 500, processor.UNEXPECTED_CHUNKING_ERROR_MESSAGE)
        self.assertNotIn("secret internal info", response.text)


if __name__ == "__main__":
    unittest.main()
