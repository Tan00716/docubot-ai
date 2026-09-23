"""Tests for app/api.py (FastAPI health check and file upload).

Every test writes uploads into a temporary folder, never into the real
storage/uploads/ folder. No Telegram, no .env, no network.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import api
from app.api import MAX_UPLOAD_SIZE_BYTES, app

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
    """Base class: point UPLOAD_DIR at a fresh temporary folder for each test."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_root = Path(temp_dir.name)
        self.upload_dir = self.temp_root / "uploads"

        upload_dir_patch = patch.object(api, "UPLOAD_DIR", self.upload_dir)
        upload_dir_patch.start()
        self.addCleanup(upload_dir_patch.stop)

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

    def test_unexpected_error_returns_generic_500(self):
        with patch.object(api.uuid, "uuid4", side_effect=RuntimeError("secret internal info")):
            response = self.upload("readme.txt", SAMPLE_TXT, "text/plain")

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error."})
        self.assertNotIn("secret internal info", response.text)
        self.assertNotIn("Traceback", response.text)


if __name__ == "__main__":
    unittest.main()
