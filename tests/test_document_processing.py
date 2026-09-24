"""Tests for app/document_processing (parsers, normalization, SQLite, pipeline).

Every test uses temporary folders and a temporary SQLite file, never the
real storage/ folder. No Telegram, no .env, no network.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import json
import logging
import sqlite3
import tempfile
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from app.document_processing import database, processor
from app.document_processing.models import (
    AlreadyProcessingError,
    DocumentContentError,
    DocumentNotFoundError,
    DocumentStatus,
    InvalidDocumentIdError,
    PROCESSED_SCHEMA_VERSION,
    ProcessedDocument,
    ProcessingError,
    Section,
    StorageError,
    StoredFileMissingError,
    UnsupportedFileTypeError,
)
from app.document_processing.parsers import (
    extract_sections,
    normalize_text,
    parse_docx,
    parse_pdf,
    parse_txt,
)
from sample_documents import (
    make_docx,
    make_encrypted_pdf,
    make_pdf,
    make_zip_without_word_document,
)

# pypdf logs warnings for broken PDFs. The tests create broken PDFs on
# purpose, so keep the test output readable.
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("app").setLevel(logging.CRITICAL)


def texts(sections):
    return [section.text for section in sections]


def locations(sections):
    return [section.source_location for section in sections]


# --- Normalization ------------------------------------------------------------


class NormalizationTests(unittest.TestCase):
    def test_windows_and_old_mac_line_endings_become_newlines(self):
        self.assertEqual(normalize_text("one\r\ntwo\rthree\nfour"), "one\ntwo\nthree\nfour")

    def test_repeated_spaces_and_tabs_become_one_space(self):
        self.assertEqual(normalize_text("Hello    world\t\tagain"), "Hello world again")

    def test_spaces_at_line_start_and_end_are_removed(self):
        self.assertEqual(normalize_text("   padded line   \n  next  "), "padded line\nnext")

    def test_single_line_breaks_are_kept(self):
        self.assertEqual(normalize_text("line one\nline two"), "line one\nline two")

    def test_paragraph_boundary_is_kept(self):
        self.assertEqual(normalize_text("Para one.\n\nPara two."), "Para one.\n\nPara two.")

    def test_extra_blank_lines_collapse_to_one_paragraph_break(self):
        self.assertEqual(normalize_text("A\n\n\n\n\nB"), "A\n\nB")

    def test_whitespace_only_lines_count_as_blank_lines(self):
        self.assertEqual(normalize_text("A\n   \n\t\nB"), "A\n\nB")

    def test_non_breaking_spaces_become_normal_spaces(self):
        self.assertEqual(normalize_text("Hello\u00a0\u00a0world"), "Hello world")

    def test_null_characters_are_removed(self):
        self.assertEqual(normalize_text("Doc\x00Bot"), "DocBot")

    def test_punctuation_and_non_ascii_text_are_preserved(self):
        text = 'Dr. Smith said: "Hello, world!" (Really?) - yes; 100% & $5.00. 你好，世界。'

        self.assertEqual(normalize_text(text), text)

    def test_whitespace_only_text_becomes_empty(self):
        self.assertEqual(normalize_text(" \n\t\r\n "), "")


# --- TXT ----------------------------------------------------------------------


class TxtParserTests(unittest.TestCase):
    def test_blocks_are_split_on_blank_lines_with_line_numbers(self):
        data = b"First line\nsecond line\n\nThird block\n"

        sections = parse_txt(data)

        self.assertEqual(texts(sections), ["First line\nsecond line", "Third block"])
        self.assertEqual(
            locations(sections),
            [
                {"block": 1, "line_start": 1, "line_end": 2},
                {"block": 2, "line_start": 4, "line_end": 4},
            ],
        )

    def test_utf8_text_is_decoded(self):
        sections = parse_txt("DocuBot 你好 café".encode("utf-8"))

        self.assertEqual(texts(sections), ["DocuBot 你好 café"])

    def test_utf8_bom_is_removed(self):
        sections = parse_txt(b"\xef\xbb\xbfHello BOM")

        self.assertEqual(texts(sections), ["Hello BOM"])

    def test_crlf_file_keeps_correct_line_numbers(self):
        sections = parse_txt(b"a\r\n\r\n\r\nb\r\n")

        self.assertEqual(
            locations(sections),
            [
                {"block": 1, "line_start": 1, "line_end": 1},
                {"block": 2, "line_start": 4, "line_end": 4},
            ],
        )

    def test_invalid_utf8_is_rejected_with_safe_message(self):
        with self.assertRaises(DocumentContentError) as caught:
            parse_txt(b"\xff\xfe\xfa not utf-8")

        self.assertEqual(caught.exception.safe_message, "The text file is not valid UTF-8.")

    def test_empty_txt_is_rejected(self):
        for data in [b"", b"   \n\n\t  \r\n"]:
            with self.subTest(data=data):
                with self.assertRaises(DocumentContentError) as caught:
                    extract_sections(data, ".txt")

                self.assertEqual(caught.exception.safe_message, "The text file contains no text.")


# --- PDF ----------------------------------------------------------------------


class PdfParserTests(unittest.TestCase):
    def test_text_pdf_is_extracted(self):
        sections = parse_pdf(make_pdf(["Hello from a PDF."]))

        self.assertEqual(texts(sections), ["Hello from a PDF."])
        self.assertEqual(locations(sections), [{"page": 1}])

    def test_multiple_pages_keep_order_and_page_numbers(self):
        sections = parse_pdf(make_pdf(["Page one.", "Page two.", "Page three."]))

        self.assertEqual(texts(sections), ["Page one.", "Page two.", "Page three."])
        self.assertEqual(locations(sections), [{"page": 1}, {"page": 2}, {"page": 3}])

    def test_page_boundaries_are_not_merged(self):
        sections = parse_pdf(make_pdf(["Alpha line 1\nAlpha line 2", "Beta"]))

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].text, "Alpha line 1\nAlpha line 2")
        self.assertNotIn("Beta", sections[0].text)

    def test_page_without_text_is_skipped_but_page_numbers_stay_real(self):
        sections = parse_pdf(make_pdf(["First", None, "Third"]))

        self.assertEqual(locations(sections), [{"page": 1}, {"page": 3}])

    def test_corrupted_pdf_is_rejected_with_safe_message(self):
        broken_inputs = [
            b"this is not a pdf",
            b"%PDF-1.4 synthetic test content",
            make_pdf(["Hello"])[:120],  # truncated file
        ]
        for data in broken_inputs:
            with self.subTest(data=data[:20]):
                with self.assertRaises(DocumentContentError) as caught:
                    parse_pdf(data)

                self.assertEqual(
                    caught.exception.safe_message,
                    "The PDF file is corrupted or cannot be read.",
                )

    def test_pdf_without_extractable_text_reports_no_machine_readable_text(self):
        with self.assertRaises(DocumentContentError) as caught:
            extract_sections(make_pdf([None, None]), ".pdf")

        self.assertIn("No machine-readable text", caught.exception.safe_message)
        self.assertIn("no OCR", caught.exception.safe_message)

    def test_encrypted_pdf_is_rejected(self):
        with self.assertRaises(DocumentContentError) as caught:
            parse_pdf(make_encrypted_pdf(["Top secret"]))

        self.assertEqual(caught.exception.safe_message, "Encrypted PDFs are not supported.")


# --- DOCX ---------------------------------------------------------------------


class DocxParserTests(unittest.TestCase):
    def test_docx_paragraph_is_extracted(self):
        sections = parse_docx(make_docx(["Hello from Word."]))

        self.assertEqual(texts(sections), ["Hello from Word."])

    def test_multiple_paragraphs_keep_document_order(self):
        sections = parse_docx(make_docx(["First", "Second", "Third"]))

        self.assertEqual(texts(sections), ["First", "Second", "Third"])
        self.assertEqual(
            locations(sections), [{"paragraph": 1}, {"paragraph": 2}, {"paragraph": 3}]
        )

    def test_paragraph_numbers_count_empty_paragraphs(self):
        sections = parse_docx(make_docx(["First", "", "Third"]))

        self.assertEqual(locations(sections), [{"paragraph": 1}, {"paragraph": 3}])

    def test_docx_never_invents_page_numbers(self):
        sections = parse_docx(make_docx(["One", "Two"]))

        for section in sections:
            self.assertNotIn("page", section.source_location)

    def test_invalid_docx_is_rejected_with_safe_message(self):
        broken_inputs = [b"PK synthetic docx test content", make_zip_without_word_document()]
        for data in broken_inputs:
            with self.subTest(data=data[:20]):
                with self.assertRaises(DocumentContentError) as caught:
                    parse_docx(data)

                self.assertEqual(
                    caught.exception.safe_message,
                    "The DOCX file is invalid or cannot be read.",
                )

    def test_docx_that_unpacks_too_large_is_rejected(self):
        data = make_docx(["Normal looking document"])

        # Real zip bombs are huge; lowering the limit tests the same check.
        with patch("app.document_processing.parsers.MAX_DOCX_UNCOMPRESSED_BYTES", 1000):
            with self.assertRaises(DocumentContentError) as caught:
                parse_docx(data)

        self.assertEqual(caught.exception.safe_message, "The DOCX file is too large to process.")

    def test_docx_within_size_limit_is_accepted(self):
        self.assertEqual(texts(parse_docx(make_docx(["Fine"]))), ["Fine"])

    def test_docx_without_text_is_rejected(self):
        with self.assertRaises(DocumentContentError):
            extract_sections(make_docx(["", "   "]), ".docx")


# --- Common representation ----------------------------------------------------


class CommonRepresentationTests(unittest.TestCase):
    def test_every_parser_returns_the_same_section_shape(self):
        samples = {
            ".txt": b"Plain text.",
            ".pdf": make_pdf(["PDF text."]),
            ".docx": make_docx(["DOCX text."]),
        }
        for extension, data in samples.items():
            with self.subTest(extension=extension):
                sections = extract_sections(data, extension)

                self.assertTrue(sections)
                for section in sections:
                    self.assertIsInstance(section, Section)
                    self.assertIsInstance(section.text, str)
                    self.assertTrue(section.source_location)
                    for key, value in section.source_location.items():
                        self.assertIsInstance(key, str)
                        self.assertIsInstance(value, int)

    def test_unsupported_extension_is_rejected(self):
        with self.assertRaises(UnsupportedFileTypeError):
            extract_sections(b"MZ...", ".exe")

    def test_processed_document_round_trips_through_dict(self):
        document = ProcessedDocument(
            document_id="a" * 32,
            source_filename="handbook.pdf",
            file_type=".pdf",
            sections=(Section({"page": 1}, "Hello"), Section({"page": 2}, "World")),
        )

        data = document.to_dict()

        self.assertEqual(data["schema_version"], PROCESSED_SCHEMA_VERSION)
        self.assertEqual(ProcessedDocument.from_dict(data), document)

    def test_unknown_schema_version_is_rejected(self):
        data = {"schema_version": 999, "document_id": "x", "source_filename": "x",
                "file_type": ".txt", "sections": []}

        with self.assertRaises(ValueError):
            ProcessedDocument.from_dict(data)


# --- SQLite -------------------------------------------------------------------


class TempStorageTestCase(unittest.TestCase):
    """Base class: fresh temporary upload/processed folders and database."""

    def setUp(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.root = Path(temp_dir.name)
        self.upload_dir = self.root / "uploads"
        self.processed_dir = self.root / "processed"
        self.db_path = self.root / "metadata" / "documents.db"

    def add_document(self, data=b"Hello DocuBot.", extension=".txt",
                     original_filename=None, write_file=True):
        document_id = uuid.uuid4().hex
        stored_filename = f"{document_id}{extension}"
        if write_file:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            (self.upload_dir / stored_filename).write_bytes(data)
        database.create_document(
            self.db_path,
            document_id=document_id,
            original_filename=original_filename or f"sample{extension}",
            stored_filename=stored_filename,
            extension=extension,
            content_type="text/plain",
            size_bytes=len(data),
        )
        return document_id

    def status_of(self, document_id):
        return database.get_document(self.db_path, document_id).status

    def run_processing(self, document_id):
        return processor.process_document(
            document_id,
            upload_dir=self.upload_dir,
            processed_dir=self.processed_dir,
            db_path=self.db_path,
        )


class FakeClock:
    """Replacement for database.utc_now that returns increasing times."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return f"2026-01-01T00:00:{self.calls:02d}.000000+00:00"


class DatabaseTests(TempStorageTestCase):
    def test_initialize_creates_database_file_and_table(self):
        database.initialize(self.db_path)

        self.assertTrue(self.db_path.is_file())
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        self.assertIn(("documents",), tables)

    def test_created_document_starts_as_uploaded(self):
        document_id = self.add_document()

        record = database.get_document(self.db_path, document_id)

        self.assertEqual(record.status, DocumentStatus.UPLOADED)
        self.assertEqual(record.original_filename, "sample.txt")
        self.assertEqual(record.created_at, record.updated_at)
        self.assertIsNone(record.processed_path)
        self.assertIsNone(record.error_message)

    def test_unknown_document_returns_none(self):
        database.initialize(self.db_path)

        self.assertIsNone(database.get_document(self.db_path, "0" * 32))

    def test_creating_same_document_twice_keeps_the_first_row(self):
        document_id = self.add_document(original_filename="first.txt")

        database.create_document(
            self.db_path, document_id=document_id, original_filename="second.txt",
            stored_filename="x.txt", extension=".txt", content_type=None, size_bytes=1,
        )

        self.assertEqual(
            database.get_document(self.db_path, document_id).original_filename, "first.txt"
        )

    def test_uploaded_to_processing(self):
        document_id = self.add_document()

        claimed = database.claim_for_processing(self.db_path, document_id)

        self.assertTrue(claimed)
        self.assertEqual(self.status_of(document_id), DocumentStatus.PROCESSING)

    def test_document_cannot_be_claimed_twice(self):
        document_id = self.add_document()
        database.claim_for_processing(self.db_path, document_id)

        self.assertFalse(database.claim_for_processing(self.db_path, document_id))

    def test_stale_processing_can_be_claimed_again(self):
        document_id = self.add_document()
        database.claim_for_processing(self.db_path, document_id)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(
                "UPDATE documents SET updated_at = '2000-01-01T00:00:00.000000+00:00'"
            )

        self.assertTrue(database.claim_for_processing(self.db_path, document_id))

    def test_processing_to_completed_stores_processed_path(self):
        document_id = self.add_document()
        database.claim_for_processing(self.db_path, document_id)

        database.mark_completed(self.db_path, document_id, f"{document_id}.json")

        record = database.get_document(self.db_path, document_id)
        self.assertEqual(record.status, DocumentStatus.COMPLETED)
        self.assertEqual(record.processed_path, f"{document_id}.json")
        self.assertIsNone(record.error_message)

    def test_processing_to_failed_stores_error_message(self):
        document_id = self.add_document()
        database.claim_for_processing(self.db_path, document_id)

        database.mark_failed(self.db_path, document_id, "The PDF file is corrupted.")

        record = database.get_document(self.db_path, document_id)
        self.assertEqual(record.status, DocumentStatus.FAILED)
        self.assertEqual(record.error_message, "The PDF file is corrupted.")
        self.assertIsNone(record.processed_path)

    def test_completed_requires_processing_first(self):
        document_id = self.add_document()

        database.mark_completed(self.db_path, document_id, "x.json")

        self.assertEqual(self.status_of(document_id), DocumentStatus.UPLOADED)

    def test_status_changes_update_updated_at_but_not_created_at(self):
        with patch.object(database, "utc_now", FakeClock()):
            document_id = self.add_document()
            created = database.get_document(self.db_path, document_id)
            database.claim_for_processing(self.db_path, document_id)
            processing = database.get_document(self.db_path, document_id)
            database.mark_completed(self.db_path, document_id, "x.json")
            completed = database.get_document(self.db_path, document_id)

        self.assertLess(created.updated_at, processing.updated_at)
        self.assertLess(processing.updated_at, completed.updated_at)
        self.assertEqual(created.created_at, completed.created_at)

    def test_database_rejects_unknown_status_values(self):
        database.initialize(self.db_path)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO documents VALUES "
                    "('id', 'a.txt', 'a.txt', '.txt', NULL, 1, 'weird', 't', 't', NULL, NULL)"
                )

    def test_sqlite_failure_becomes_safe_storage_error(self):
        # A folder cannot be opened as a database file.
        with self.assertRaises(StorageError) as caught:
            database.get_document(self.root, "0" * 32)

        self.assertEqual(caught.exception.safe_message, "Could not access the document database.")
        self.assertNotIn(str(self.root), caught.exception.safe_message)


# --- Pipeline -----------------------------------------------------------------


class ProcessingSuccessTests(TempStorageTestCase):
    def read_output(self, document_id):
        return json.loads((self.processed_dir / f"{document_id}.json").read_text("utf-8"))

    def test_txt_document_is_processed_and_completed(self):
        document_id = self.add_document(b"Hello   DocuBot.\r\n\r\nSecond block.", ".txt",
                                        original_filename="notes.txt")

        result = self.run_processing(document_id)

        self.assertEqual(result.record.status, DocumentStatus.COMPLETED)
        self.assertEqual(result.record.processed_path, f"{document_id}.json")
        self.assertIsNone(result.record.error_message)
        self.assertEqual(
            self.read_output(document_id),
            {
                "schema_version": PROCESSED_SCHEMA_VERSION,
                "document_id": document_id,
                "source_filename": "notes.txt",
                "file_type": ".txt",
                "sections": [
                    {"source_location": {"block": 1, "line_start": 1, "line_end": 1},
                     "text": "Hello DocuBot."},
                    {"source_location": {"block": 2, "line_start": 3, "line_end": 3},
                     "text": "Second block."},
                ],
            },
        )

    def test_pdf_output_keeps_page_numbers(self):
        document_id = self.add_document(make_pdf(["One", None, "Three"]), ".pdf")

        self.run_processing(document_id)

        sections = self.read_output(document_id)["sections"]
        self.assertEqual(
            sections,
            [{"source_location": {"page": 1}, "text": "One"},
             {"source_location": {"page": 3}, "text": "Three"}],
        )

    def test_docx_output_keeps_paragraph_order(self):
        document_id = self.add_document(make_docx(["Intro", "Body", "End"]), ".docx")

        self.run_processing(document_id)

        sections = self.read_output(document_id)["sections"]
        self.assertEqual([s["text"] for s in sections], ["Intro", "Body", "End"])
        self.assertEqual(sections[2]["source_location"], {"paragraph": 3})

    def test_status_is_processing_while_the_parser_runs(self):
        document_id = self.add_document()
        seen_statuses = []

        def fake_extract(data, extension):
            seen_statuses.append(self.status_of(document_id))
            return [Section({"block": 1}, "text")]

        with patch.object(processor, "extract_sections", side_effect=fake_extract):
            self.run_processing(document_id)

        self.assertEqual(seen_statuses, [DocumentStatus.PROCESSING])
        self.assertEqual(self.status_of(document_id), DocumentStatus.COMPLETED)

    def test_output_is_deterministic(self):
        document_id = self.add_document(make_pdf(["Same", "Output"]), ".pdf")
        output_file = self.processed_dir / f"{document_id}.json"

        self.run_processing(document_id)
        first = output_file.read_bytes()
        self.run_processing(document_id)
        second = output_file.read_bytes()

        self.assertEqual(first, second)
        self.assertNotIn(b"\r\n", first)

    def test_non_ascii_text_is_stored_readably(self):
        document_id = self.add_document("你好，DocuBot。".encode("utf-8"))

        self.run_processing(document_id)

        raw = (self.processed_dir / f"{document_id}.json").read_text("utf-8")
        self.assertIn("你好，DocuBot。", raw)

    def test_only_expected_files_are_written(self):
        document_id = self.add_document()

        self.run_processing(document_id)

        written = sorted(p.relative_to(self.root).as_posix()
                         for p in self.root.rglob("*") if p.is_file())
        self.assertEqual(
            written,
            sorted([f"uploads/{document_id}.txt", f"processed/{document_id}.json",
                    "metadata/documents.db"]),
        )

    def test_failed_document_can_be_processed_again(self):
        document_id = self.add_document(b"\xff\xfe bad bytes")
        with self.assertRaises(DocumentContentError):
            self.run_processing(document_id)

        (self.upload_dir / f"{document_id}.txt").write_bytes(b"Fixed text.")
        result = self.run_processing(document_id)

        self.assertEqual(result.record.status, DocumentStatus.COMPLETED)
        self.assertIsNone(result.record.error_message)

    def test_legacy_upload_without_record_is_registered_and_processed(self):
        document_id = uuid.uuid4().hex
        self.upload_dir.mkdir(parents=True)
        (self.upload_dir / f"{document_id}.txt").write_bytes(b"Uploaded before Batch 3.")

        result = self.run_processing(document_id)

        self.assertEqual(result.record.status, DocumentStatus.COMPLETED)
        self.assertEqual(result.record.original_filename, f"{document_id}.txt")

    def test_processed_document_can_be_loaded_for_later_stages(self):
        document_id = self.add_document(make_pdf(["Load me"]), ".pdf")
        result = self.run_processing(document_id)

        loaded = processor.load_processed_document(result.record, self.processed_dir)

        self.assertEqual(loaded, result.document)

    def test_loading_unprocessed_document_is_refused(self):
        document_id = self.add_document()
        record = database.get_document(self.db_path, document_id)

        with self.assertRaises(ProcessingError):
            processor.load_processed_document(record, self.processed_dir)


class ProcessingFailureTests(TempStorageTestCase):
    def assert_failed_with(self, document_id, message):
        record = database.get_document(self.db_path, document_id)
        self.assertEqual(record.status, DocumentStatus.FAILED)
        self.assertEqual(record.error_message, message)
        self.assertIsNone(record.processed_path)
        self.assertFalse((self.processed_dir / f"{document_id}.json").exists())

    def test_corrupted_pdf_marks_document_failed(self):
        document_id = self.add_document(b"%PDF-1.4 broken", ".pdf")

        with self.assertRaises(DocumentContentError):
            self.run_processing(document_id)

        self.assert_failed_with(document_id, "The PDF file is corrupted or cannot be read.")

    def test_image_only_pdf_marks_document_failed_with_clear_message(self):
        document_id = self.add_document(make_pdf([None]), ".pdf")

        with self.assertRaises(DocumentContentError) as caught:
            self.run_processing(document_id)

        self.assertIn("No machine-readable text", caught.exception.safe_message)
        self.assert_failed_with(document_id, caught.exception.safe_message)

    def test_invalid_docx_marks_document_failed(self):
        document_id = self.add_document(b"PK not a docx", ".docx")

        with self.assertRaises(DocumentContentError):
            self.run_processing(document_id)

        self.assert_failed_with(document_id, "The DOCX file is invalid or cannot be read.")

    def test_invalid_utf8_marks_document_failed(self):
        document_id = self.add_document(b"\xff\xfe\xfa")

        with self.assertRaises(DocumentContentError):
            self.run_processing(document_id)

        self.assert_failed_with(document_id, "The text file is not valid UTF-8.")

    def test_empty_document_marks_document_failed(self):
        document_id = self.add_document(b"   \n\n  ")

        with self.assertRaises(DocumentContentError):
            self.run_processing(document_id)

        self.assert_failed_with(document_id, "The text file contains no text.")

    def test_missing_stored_file_marks_document_failed(self):
        document_id = self.add_document(write_file=False)

        with self.assertRaises(StoredFileMissingError):
            self.run_processing(document_id)

        self.assert_failed_with(document_id, "The stored file for this document is missing.")

    def test_unsupported_file_type_marks_document_failed(self):
        document_id = self.add_document(b"MZ", ".exe")

        with self.assertRaises(UnsupportedFileTypeError):
            self.run_processing(document_id)

        self.assert_failed_with(document_id, UnsupportedFileTypeError.safe_message)

    def test_unexpected_parser_crash_stores_only_a_generic_message(self):
        document_id = self.add_document()
        crash = RuntimeError("secret detail C:\\internal\\path")

        with patch.object(processor, "extract_sections", side_effect=crash):
            with self.assertRaises(ProcessingError) as caught:
                self.run_processing(document_id)

        self.assertEqual(caught.exception.safe_message, processor.UNEXPECTED_ERROR_MESSAGE)
        self.assert_failed_with(document_id, processor.UNEXPECTED_ERROR_MESSAGE)
        record = database.get_document(self.db_path, document_id)
        self.assertNotIn("secret", record.error_message)
        self.assertNotIn("Traceback", record.error_message)

    def test_output_write_failure_marks_document_failed(self):
        document_id = self.add_document()
        blocker = self.root / "blocker"
        blocker.write_text("a file, not a folder")
        self.processed_dir = blocker / "processed"

        with self.assertRaises(StorageError):
            self.run_processing(document_id)

        self.assert_failed_with(document_id, "Could not save the processed document.")

    def test_sqlite_failure_while_completing_does_not_report_success(self):
        document_id = self.add_document()

        with patch.object(database, "mark_completed", side_effect=StorageError()):
            with self.assertRaises(StorageError):
                self.run_processing(document_id)

        self.assertEqual(self.status_of(document_id), DocumentStatus.FAILED)

    def test_unknown_document_is_not_found(self):
        database.initialize(self.db_path)

        with self.assertRaises(DocumentNotFoundError):
            self.run_processing(uuid.uuid4().hex)

    def test_invalid_document_ids_are_rejected_before_any_file_access(self):
        for bad_id in ["../../etc/passwd", "..", "ABCDEF" * 6, "a" * 31, "a" * 32 + ".txt", ""]:
            with self.subTest(document_id=bad_id):
                with self.assertRaises(InvalidDocumentIdError):
                    self.run_processing(bad_id)

        self.assertFalse(self.db_path.exists())

    def test_document_already_processing_is_rejected_and_left_unchanged(self):
        document_id = self.add_document()
        database.claim_for_processing(self.db_path, document_id)

        with self.assertRaises(AlreadyProcessingError):
            self.run_processing(document_id)

        self.assertEqual(self.status_of(document_id), DocumentStatus.PROCESSING)


class PathSafetyTests(TempStorageTestCase):
    def test_safe_child_path_rejects_paths_outside_the_folder(self):
        for name in ["../escape.json", "..\\escape.json", "sub/inner.json", "/abs.json"]:
            with self.subTest(name=name):
                with self.assertRaises(StorageError):
                    processor.safe_child_path(self.processed_dir, name)

    def test_safe_child_path_accepts_a_plain_filename(self):
        path = processor.safe_child_path(self.processed_dir, "abc.json")

        self.assertEqual(path, (self.processed_dir / "abc.json").resolve())


if __name__ == "__main__":
    unittest.main()
