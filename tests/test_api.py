"""Tests for app/api.py (health check, file upload, document processing).

Every test writes uploads, processed output and the SQLite database into
temporary folders, never into the real storage/ folder.
No Telegram, no .env, no network.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import logging
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import api
from app.api import MAX_UPLOAD_SIZE_BYTES, app
from app.document_processing import database, processor
from app.document_processing.models import StorageError
from sample_documents import make_docx, make_pdf

# Broken documents are uploaded on purpose; keep library/app logs quiet.
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("app").setLevel(logging.CRITICAL)

# Server-side stored names look like: 32 hex characters + extension.
STORED_NAME_PATTERN = re.compile(r"^[0-9a-f]{32}\.(pdf|docx|txt)$")

# Tiny synthetic file contents. The API does not parse files yet, so these
# only need to be non-empty bytes; they are not real PDF/DOCX documents.
SAMPLE_PDF = b"%PDF-1.4 synthetic test content"
SAMPLE_DOCX = b"PK synthetic docx test content"
SAMPLE_TXT = "Hello DocuBot 你好".encode("utf-8")

DOCX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


class ApiTestCase(unittest.TestCase):
    """Base class: point all storage paths at fresh temporary folders."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_root = Path(temp_dir.name)
        self.upload_dir = self.temp_root / "uploads"
        self.processed_dir = self.temp_root / "processed"
        self.db_path = self.temp_root / "metadata" / "documents.db"

        for name, value in [
            ("UPLOAD_DIR", self.upload_dir),
            ("PROCESSED_DIR", self.processed_dir),
            ("DATABASE_PATH", self.db_path),
        ]:
            path_patch = patch.object(api, name, value)
            path_patch.start()
            self.addCleanup(path_patch.stop)

        # raise_server_exceptions=False lets us see the real 500 response
        # that an API user would receive, instead of a Python exception.
        self.client = TestClient(app, raise_server_exceptions=False)

    def upload(self, filename, content, content_type="application/octet-stream"):
        return self.client.post(
            "/upload", files={"file": (filename, content, content_type)}
        )

    def stored_files(self):
        if not self.upload_dir.exists():
            return []
        return list(self.upload_dir.iterdir())


class HealthTests(ApiTestCase):
    def test_health_returns_200(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

    def test_health_returns_status_ok(self):
        response = self.client.get("/health")

        self.assertEqual(response.json(), {"status": "ok"})


class SuccessfulUploadTests(ApiTestCase):
    def test_pdf_upload_succeeds(self):
        response = self.upload("handbook.pdf", SAMPLE_PDF, "application/pdf")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["extension"], ".pdf")

    def test_docx_upload_succeeds(self):
        response = self.upload("notes.docx", SAMPLE_DOCX, DOCX_CONTENT_TYPE)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["extension"], ".docx")

    def test_txt_upload_succeeds(self):
        response = self.upload("readme.txt", SAMPLE_TXT, "text/plain")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["extension"], ".txt")

    def test_extension_check_is_case_insensitive(self):
        response = self.upload("REPORT.PDF", SAMPLE_PDF, "application/pdf")

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["extension"], ".pdf")

    def test_response_contains_expected_metadata(self):
        response = self.upload("handbook.pdf", SAMPLE_PDF, "application/pdf")
        body = response.json()

        self.assertEqual(
            set(body),
            {
                "file_id",
                "filename",
                "stored_filename",
                "extension",
                "content_type",
                "size_bytes",
                "status",
            },
        )
        self.assertRegex(body["file_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(body["filename"], "handbook.pdf")
        self.assertEqual(body["stored_filename"], f"{body['file_id']}.pdf")
        self.assertEqual(body["content_type"], "application/pdf")
        self.assertEqual(body["size_bytes"], len(SAMPLE_PDF))
        self.assertEqual(body["status"], "stored")

    def test_response_does_not_expose_server_paths(self):
        response = self.upload("handbook.pdf", SAMPLE_PDF, "application/pdf")

        self.assertNotIn(str(self.temp_root), response.text)
        self.assertNotIn("storage", response.text)

    def test_uploaded_file_is_written_to_upload_dir(self):
        response = self.upload("readme.txt", SAMPLE_TXT, "text/plain")
        stored_filename = response.json()["stored_filename"]

        saved_file = self.upload_dir / stored_filename
        self.assertTrue(saved_file.is_file())
        self.assertEqual(saved_file.read_bytes(), SAMPLE_TXT)
        self.assertEqual(self.stored_files(), [saved_file])

    def test_raw_user_filename_is_not_used_as_storage_name(self):
        response = self.upload("my private report.pdf", SAMPLE_PDF, "application/pdf")
        stored_filename = response.json()["stored_filename"]

        self.assertRegex(stored_filename, STORED_NAME_PATTERN)
        self.assertFalse((self.upload_dir / "my private report.pdf").exists())

    def test_same_filename_twice_creates_two_separate_files(self):
        first = self.upload("same.txt", b"first", "text/plain").json()
        second = self.upload("same.txt", b"second", "text/plain").json()

        self.assertNotEqual(first["file_id"], second["file_id"])
        self.assertEqual(len(self.stored_files()), 2)

    def test_file_exactly_at_size_limit_is_accepted(self):
        with patch.object(api, "MAX_UPLOAD_SIZE_BYTES", 10):
            response = self.upload("limit.txt", b"x" * 10, "text/plain")

        self.assertEqual(response.status_code, 201)


class RejectedUploadTests(ApiTestCase):
    def assert_rejected(self, response, status_code):
        self.assertEqual(response.status_code, status_code)
        self.assertIn("detail", response.json())
        self.assertEqual(self.stored_files(), [], "nothing should be saved")

    def test_unsupported_extension_is_rejected(self):
        for filename in ["virus.exe", "script.py", "page.html", "archive.zip"]:
            with self.subTest(filename=filename):
                response = self.upload(filename, b"data")

                self.assert_rejected(response, 415)

    def test_filename_without_extension_is_rejected(self):
        response = self.upload("README", b"data")

        self.assert_rejected(response, 415)

    def test_file_larger_than_limit_is_rejected(self):
        too_big = b"x" * (MAX_UPLOAD_SIZE_BYTES + 1)

        response = self.upload("big.txt", too_big, "text/plain")

        self.assert_rejected(response, 413)

    def test_empty_file_is_rejected(self):
        response = self.upload("empty.txt", b"", "text/plain")

        self.assert_rejected(response, 400)

    def test_path_traversal_filenames_are_rejected(self):
        dangerous_names = [
            "../../evil.txt",
            "..\\..\\evil.txt",
            "/etc/evil.txt",
            "folder/evil.txt",
            "..",
        ]
        for filename in dangerous_names:
            with self.subTest(filename=filename):
                response = self.upload(filename, b"data", "text/plain")

                self.assert_rejected(response, 400)

        # Nothing was written next to (outside) the upload folder either.
        self.assertFalse((self.temp_root / "evil.txt").exists())

    def test_too_long_filename_is_rejected(self):
        response = self.upload("a" * 300 + ".txt", b"data", "text/plain")

        self.assert_rejected(response, 400)

    def test_missing_file_returns_422(self):
        response = self.client.post("/upload")

        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.stored_files(), [])


class ServerErrorTests(ApiTestCase):
    def test_storage_write_failure_returns_500_without_details(self):
        # Make UPLOAD_DIR impossible to create: its parent is a regular file.
        blocker = self.temp_root / "not-a-folder"
        blocker.write_text("blocker")
        with patch.object(api, "UPLOAD_DIR", blocker / "uploads"):
            response = self.upload("readme.txt", SAMPLE_TXT, "text/plain")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Could not store the file."})
        self.assertNotIn(str(self.temp_root), response.text)

    def test_metadata_failure_returns_500_and_removes_stored_file(self):
        with patch.object(api.database, "create_document", side_effect=StorageError()):
            response = self.upload("readme.txt", SAMPLE_TXT, "text/plain")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Could not store the file."})
        self.assertEqual(self.stored_files(), [], "an untracked file must not remain")

    def test_unexpected_error_returns_generic_500(self):
        with patch.object(api.uuid, "uuid4", side_effect=RuntimeError("secret internal info")):
            response = self.upload("readme.txt", SAMPLE_TXT, "text/plain")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error."})
        self.assertNotIn("secret internal info", response.text)
        self.assertNotIn("Traceback", response.text)



# --- Document processing endpoints ---------------------------------------------


class DocumentApiTestCase(ApiTestCase):
    def upload_document(self, filename, content):
        response = self.upload(filename, content)
        self.assertEqual(response.status_code, 201)
        return response.json()["file_id"]

    def process(self, document_id):
        return self.client.post(f"/documents/{document_id}/process")

    def get_document(self, document_id):
        return self.client.get(f"/documents/{document_id}")


class DocumentMetadataTests(DocumentApiTestCase):
    def test_upload_creates_uploaded_record(self):
        document_id = self.upload_document("handbook.txt", SAMPLE_TXT)

        response = self.get_document(document_id)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["document_id"], document_id)
        self.assertEqual(body["original_filename"], "handbook.txt")
        self.assertEqual(body["file_type"], ".txt")
        self.assertEqual(body["size_bytes"], len(SAMPLE_TXT))
        self.assertEqual(body["status"], "uploaded")
        self.assertIsNone(body["processed_path"])
        self.assertIsNone(body["error_message"])

    def test_metadata_never_contains_text_or_server_paths(self):
        document_id = self.upload_document("secret-notes.txt", b"Very private content")
        self.process(document_id)

        response = self.get_document(document_id)

        self.assertNotIn("Very private content", response.text)
        self.assertNotIn(str(self.temp_root), response.text)
        self.assertNotIn("text", response.json())
        self.assertNotIn("sections", response.json())

    def test_unknown_document_returns_404(self):
        response = self.get_document("0" * 32)

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Document not found."})

    def test_invalid_document_id_returns_400(self):
        for bad_id in ["not-a-real-id", "%2e%2e", "A" * 32]:
            with self.subTest(document_id=bad_id):
                response = self.get_document(bad_id)

                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json(), {"detail": "Invalid document ID."})

    def test_database_failure_returns_safe_500(self):
        # A folder cannot be opened as a SQLite database file.
        with patch.object(api, "DATABASE_PATH", self.temp_root):
            response = self.get_document("0" * 32)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json(), {"detail": "Could not access the document database."}
        )
        self.assertNotIn(str(self.temp_root), response.text)


class ProcessSuccessTests(DocumentApiTestCase):
    def test_txt_is_processed(self):
        document_id = self.upload_document("notes.txt", b"First block.\n\nSecond block.")

        response = self.process(document_id)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["section_count"], 2)
        self.assertEqual(body["processed_path"], f"{document_id}.json")
        self.assertIsNone(body["error_message"])
        self.assertTrue((self.processed_dir / f"{document_id}.json").is_file())

    def test_pdf_is_processed_with_one_section_per_text_page(self):
        document_id = self.upload_document("guide.pdf", make_pdf(["One", "Two", None]))

        response = self.process(document_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["section_count"], 2)

    def test_docx_is_processed(self):
        document_id = self.upload_document("memo.docx", make_docx(["Hello", "World"]))

        response = self.process(document_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["file_type"], ".docx")
        self.assertEqual(response.json()["section_count"], 2)

    def test_status_endpoint_shows_completed_after_processing(self):
        document_id = self.upload_document("notes.txt", SAMPLE_TXT)
        self.process(document_id)

        body = self.get_document(document_id).json()

        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["processed_path"], f"{document_id}.json")

    def test_processing_twice_is_allowed(self):
        document_id = self.upload_document("notes.txt", SAMPLE_TXT)

        self.assertEqual(self.process(document_id).status_code, 200)
        self.assertEqual(self.process(document_id).status_code, 200)


class ProcessFailureTests(DocumentApiTestCase):
    def assert_failed(self, response, status_code, message):
        self.assertEqual(response.status_code, status_code)
        self.assertEqual(response.json(), {"detail": message})
        self.assertNotIn(str(self.temp_root), response.text)
        self.assertNotIn("Traceback", response.text)

    def assert_recorded_failure(self, document_id, message):
        body = self.get_document(document_id).json()
        self.assertEqual(body["status"], "failed")
        self.assertEqual(body["error_message"], message)
        self.assertIsNone(body["processed_path"])

    def test_unknown_document_returns_404(self):
        response = self.process("0" * 32)

        self.assert_failed(response, 404, "Document not found.")

    def test_invalid_document_id_returns_400(self):
        response = self.process("not-a-valid-id")

        self.assert_failed(response, 400, "Invalid document ID.")

    def test_corrupted_pdf_returns_422_and_is_marked_failed(self):
        document_id = self.upload_document("broken.pdf", SAMPLE_PDF)

        response = self.process(document_id)

        message = "The PDF file is corrupted or cannot be read."
        self.assert_failed(response, 422, message)
        self.assert_recorded_failure(document_id, message)

    def test_image_only_pdf_returns_no_machine_readable_text(self):
        document_id = self.upload_document("scan.pdf", make_pdf([None]))

        response = self.process(document_id)

        self.assertEqual(response.status_code, 422)
        self.assertIn("No machine-readable text", response.json()["detail"])
        self.assertEqual(self.get_document(document_id).json()["status"], "failed")

    def test_invalid_docx_returns_422(self):
        document_id = self.upload_document("fake.docx", SAMPLE_DOCX)

        response = self.process(document_id)

        self.assert_failed(response, 422, "The DOCX file is invalid or cannot be read.")

    def test_invalid_utf8_returns_422(self):
        document_id = self.upload_document("latin1.txt", "café".encode("latin-1"))

        response = self.process(document_id)

        message = "The text file is not valid UTF-8."
        self.assert_failed(response, 422, message)
        self.assert_recorded_failure(document_id, message)

    def test_missing_stored_file_returns_404_and_is_marked_failed(self):
        document_id = self.upload_document("gone.txt", SAMPLE_TXT)
        (self.upload_dir / f"{document_id}.txt").unlink()

        response = self.process(document_id)

        message = "The stored file for this document is missing."
        self.assert_failed(response, 404, message)
        self.assert_recorded_failure(document_id, message)

    def test_document_already_processing_returns_409(self):
        document_id = self.upload_document("busy.txt", SAMPLE_TXT)
        database.claim_for_processing(self.db_path, document_id)

        response = self.process(document_id)

        self.assert_failed(response, 409, "This document is already being processed.")
        self.assertEqual(self.get_document(document_id).json()["status"], "processing")

    def test_output_storage_failure_returns_500_and_is_marked_failed(self):
        document_id = self.upload_document("notes.txt", SAMPLE_TXT)
        blocker = self.temp_root / "blocker"
        blocker.write_text("a file, not a folder")

        with patch.object(api, "PROCESSED_DIR", blocker / "processed"):
            response = self.process(document_id)

        message = "Could not save the processed document."
        self.assert_failed(response, 500, message)
        self.assert_recorded_failure(document_id, message)

    def test_unexpected_parser_error_returns_generic_500(self):
        document_id = self.upload_document("notes.txt", SAMPLE_TXT)
        crash = RuntimeError("secret internal info")

        with patch.object(processor, "extract_sections", side_effect=crash):
            response = self.process(document_id)

        self.assert_failed(response, 500, processor.UNEXPECTED_ERROR_MESSAGE)
        self.assertNotIn("secret internal info", response.text)
        self.assert_recorded_failure(document_id, processor.UNEXPECTED_ERROR_MESSAGE)


if __name__ == "__main__":
    unittest.main()
