"""Tests for storing chunks in SQLite and for the chunking pipeline.

Every test uses temporary folders and a temporary SQLite file, never the
real storage/ folder. No Telegram, no .env, no network, no LLM.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import json
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch

from app.document_processing import database, processor
from app.document_processing.chunking import EMPTY_DOCUMENT_MESSAGE, ChunkingConfig
from app.document_processing.models import (
    ChunkingConflictError,
    ChunkingStatus,
    DocumentContentError,
    DocumentNotFoundError,
    DocumentNotProcessedError,
    DocumentStatus,
    InvalidDocumentIdError,
    PROCESSED_SCHEMA_VERSION,
    ProcessedOutputMissingError,
    ProcessingError,
    StorageError,
)
from sample_documents import make_docx, make_pdf
from test_chunking import ChunkInvariantsMixin, words
from test_document_processing import TempStorageTestCase

class ChunkingStorageTestCase(TempStorageTestCase):
    """Adds helpers to create processed documents and chunk them."""

    def processed_document(self, data=b"Hello DocuBot.\n\nSecond block.", extension=".txt"):
        document_id = self.add_document(data, extension)
        self.run_processing(document_id)
        return document_id

    def run_chunking(self, document_id, config=None):
        return processor.chunk_document(
            document_id,
            upload_dir=self.upload_dir,
            processed_dir=self.processed_dir,
            db_path=self.db_path,
            config=config or ChunkingConfig(),
        )

    def stored_chunks(self, document_id):
        return database.get_chunk_set(self.db_path, document_id)

    def chunk_row_count(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def write_processed_json(self, document_id, content):
        (self.processed_dir / f"{document_id}.json").write_text(content, encoding="utf-8")


class ChunkDatabaseTests(ChunkingStorageTestCase):
    def test_initialize_creates_chunks_table(self):
        database.initialize(self.db_path)

        with closing(sqlite3.connect(self.db_path)) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        self.assertIn(("chunks",), tables)

    def test_chunks_are_stored_and_read_back_identically(self):
        document_id = self.processed_document(words(600).encode())

        result = self.run_chunking(document_id)
        stored = self.stored_chunks(document_id)

        self.assertGreater(len(result.chunks), 1)
        self.assertEqual(stored.chunks, result.chunks)
        self.assertEqual(stored.created_at, result.created_at)
        self.assertEqual(stored.status, ChunkingStatus.CHUNKED)
        self.assertEqual(self.chunk_row_count(), len(result.chunks))

    def test_chunks_are_returned_in_chunk_index_order(self):
        document_id = self.processed_document(words(900).encode())
        self.run_chunking(document_id)

        indexes = [c.chunk_index for c in self.stored_chunks(document_id).chunks]

        self.assertEqual(indexes, list(range(len(indexes))))

    def test_document_without_chunks_is_not_chunked(self):
        document_id = self.processed_document()

        chunk_set = self.stored_chunks(document_id)

        self.assertEqual(chunk_set.chunks, ())
        self.assertIsNone(chunk_set.created_at)
        self.assertEqual(chunk_set.status, ChunkingStatus.NOT_CHUNKED)

    def test_chunking_twice_does_not_duplicate_chunks(self):
        document_id = self.processed_document(words(900).encode())

        first = self.run_chunking(document_id)
        second = self.run_chunking(document_id)

        self.assertEqual([c.chunk_id for c in first.chunks], [c.chunk_id for c in second.chunks])
        self.assertEqual(self.chunk_row_count(), len(first.chunks))
        self.assertEqual(self.stored_chunks(document_id).chunks, second.chunks)

    def test_chunking_with_new_configuration_replaces_old_chunks(self):
        document_id = self.processed_document(words(900).encode())
        self.run_chunking(document_id, ChunkingConfig(1200, 200))

        new = self.run_chunking(document_id, ChunkingConfig(300, 0))

        stored = self.stored_chunks(document_id).chunks
        self.assertEqual(stored, new.chunks)
        self.assertTrue(all(c.chunk_size == 300 for c in stored))
        self.assertEqual(self.chunk_row_count(), len(new.chunks))

    def test_chunks_of_different_documents_are_kept_apart(self):
        first_id = self.processed_document(b"Same text.")
        second_id = self.processed_document(b"Same text.")

        self.run_chunking(first_id)
        self.run_chunking(second_id)

        self.assertEqual(len(self.stored_chunks(first_id).chunks), 1)
        self.assertEqual(len(self.stored_chunks(second_id).chunks), 1)
        self.assertNotEqual(self.stored_chunks(first_id).chunks[0].chunk_id,
                            self.stored_chunks(second_id).chunks[0].chunk_id)

    def test_database_rejects_duplicate_chunk_index(self):
        document_id = self.processed_document()
        self.run_chunking(document_id)

        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO chunks VALUES ('other-id', ?, 0, 'x', 1, '[]', 1, 1, 0, 't')",
                    (document_id,),
                )

    def test_database_rejects_chunks_of_unknown_documents(self):
        database.initialize(self.db_path)

        with self.assertRaises(StorageError):
            with database.connect(self.db_path) as connection:
                connection.execute(
                    "INSERT INTO chunks VALUES ('id', ?, 0, 'x', 1, '[]', 1, 1, 0, 't')",
                    ("f" * 32,),
                )

    def test_failed_write_keeps_the_previous_chunks(self):
        document_id = self.processed_document(words(600).encode())
        before = self.run_chunking(document_id)
        record = database.get_document(self.db_path, document_id)
        duplicated = [before.chunks[0], before.chunks[0]]  # same chunk_id twice

        with self.assertRaises(StorageError):
            database.replace_chunks(self.db_path, document_id, duplicated,
                                    expected_updated_at=record.updated_at)

        self.assertEqual(self.stored_chunks(document_id).chunks, before.chunks)

    def test_processing_again_deletes_the_old_chunks(self):
        document_id = self.processed_document()
        self.run_chunking(document_id)

        self.run_processing(document_id)

        self.assertEqual(self.stored_chunks(document_id).status, ChunkingStatus.NOT_CHUNKED)
        self.assertEqual(self.chunk_row_count(), 0)

    def test_stale_document_version_is_refused_and_nothing_is_saved(self):
        document_id = self.processed_document()
        before = self.run_chunking(document_id)

        with self.assertRaises(ChunkingConflictError):
            database.replace_chunks(self.db_path, document_id, before.chunks,
                                    expected_updated_at="2000-01-01T00:00:00.000000+00:00")

        self.assertEqual(self.stored_chunks(document_id).chunks, before.chunks)

    def test_corrupted_stored_chunks_are_reported_safely(self):
        corruptions = [
            "UPDATE chunks SET source_locations = 'not json' WHERE document_id = ?",
            "UPDATE chunks SET source_locations = '[]' WHERE document_id = ?",
            "UPDATE chunks SET source_locations = '[{\"page\": \"two\"}]' WHERE document_id = ?",
            "UPDATE chunks SET char_count = char_count + 1 WHERE document_id = ?",
            # A BLOB keeps its type in a TEXT column (a number would become text).
            "UPDATE chunks SET text = CAST('abc' AS BLOB) WHERE document_id = ?",
            "DELETE FROM chunks WHERE chunk_index = 0 AND document_id = ?",
        ]
        for sql in corruptions:
            with self.subTest(sql=sql):
                document_id = self.processed_document(words(600).encode())
                self.run_chunking(document_id)
                with closing(sqlite3.connect(self.db_path)) as connection, connection:
                    connection.execute(sql, (document_id,))

                with self.assertRaises(StorageError) as caught:
                    self.stored_chunks(document_id)

                self.assertEqual(caught.exception.safe_message, database.CORRUPTED_CHUNKS_MESSAGE)


class ChunkPipelineTests(ChunkingStorageTestCase, ChunkInvariantsMixin):
    def test_txt_chunks_keep_block_and_line_locations(self):
        document_id = self.processed_document(b"First block\nline two\n\nSecond block")

        chunk = self.run_chunking(document_id).chunks[0]

        self.assertEqual(
            list(chunk.source_locations),
            [
                {"block": 1, "line_start": 1, "line_end": 2, "char_start": 0, "char_end": 20},
                {"block": 2, "line_start": 4, "line_end": 4, "char_start": 0, "char_end": 12},
            ],
        )
        self.assertEqual(chunk.text, "First block\nline two\n\nSecond block")

    def test_pdf_chunks_keep_real_page_numbers(self):
        document_id = self.processed_document(make_pdf(["Page one.", None, "Page three."]), ".pdf")

        chunk = self.run_chunking(document_id).chunks[0]

        self.assertEqual([loc["page"] for loc in chunk.source_locations], [1, 3])
        self.assertEqual(chunk.file_type, ".pdf")

    def test_pdf_pages_can_go_to_separate_chunks(self):
        pages = ["Alpha " * 30, "Beta " * 30]
        document_id = self.processed_document(make_pdf(pages), ".pdf")

        chunks = self.run_chunking(document_id, ChunkingConfig(200, 0)).chunks

        self.assertEqual([[loc["page"] for loc in c.source_locations] for c in chunks], [[1], [2]])

    def test_docx_chunks_keep_paragraph_numbers(self):
        document_id = self.processed_document(make_docx(["Intro", "", "Body"]), ".docx")

        chunk = self.run_chunking(document_id).chunks[0]

        self.assertEqual([loc["paragraph"] for loc in chunk.source_locations], [1, 3])
        for location in chunk.source_locations:
            self.assertNotIn("page", location)

    def test_real_processed_document_keeps_all_invariants(self):
        text = "\n\n".join(words(n * 20, f"p{n}w") for n in range(1, 15)).encode()
        document_id = self.processed_document(text)
        record = database.get_document(self.db_path, document_id)
        document = processor.load_processed_document(record, self.processed_dir)
        config = ChunkingConfig(500, 100)

        chunks = self.run_chunking(document_id, config).chunks

        self.assert_valid_chunks(document, list(chunks), config)

    def test_chunking_does_not_change_the_processing_status(self):
        document_id = self.processed_document()
        before = database.get_document(self.db_path, document_id)

        self.run_chunking(document_id)

        self.assertEqual(database.get_document(self.db_path, document_id), before)

    def test_chunking_writes_no_files(self):
        document_id = self.processed_document()

        self.run_chunking(document_id)

        written = sorted(p.relative_to(self.root).as_posix()
                         for p in self.root.rglob("*") if p.is_file())
        self.assertEqual(written, sorted([f"uploads/{document_id}.txt",
                                          f"processed/{document_id}.json",
                                          "metadata/documents.db"]))

    def test_document_must_be_processed_first(self):
        uploaded_id = self.add_document()
        failed_id = self.add_document(b"\xff\xfe not utf-8")
        with self.assertRaises(DocumentContentError):
            self.run_processing(failed_id)
        busy_id = self.add_document()
        database.claim_for_processing(self.db_path, busy_id)

        for document_id in [uploaded_id, failed_id, busy_id]:
            with self.subTest(status=self.status_of(document_id)):
                with self.assertRaises(DocumentNotProcessedError) as caught:
                    self.run_chunking(document_id)

                self.assertEqual(caught.exception.safe_message,
                                 "This document has not been processed yet.")
        self.assertEqual(self.chunk_row_count(), 0)

    def test_missing_processed_file_is_reported(self):
        document_id = self.processed_document()
        (self.processed_dir / f"{document_id}.json").unlink()

        with self.assertRaises(ProcessedOutputMissingError):
            self.run_chunking(document_id)

    def test_malformed_processed_json_is_rejected_safely(self):
        valid_section = {"source_location": {"block": 1}, "text": "ok"}

        def document_json(**changes):
            data = {"schema_version": PROCESSED_SCHEMA_VERSION, "source_filename": "a.txt",
                    "file_type": ".txt", "sections": [valid_section]}
            return data | changes

        cases = {
            "not json": "{ this is not json",
            "not an object": "[1, 2, 3]",
            "wrong schema version": document_json(schema_version=999),
            "sections not a list": document_json(sections="oops"),
            "text not a string": document_json(sections=[{"source_location": {"block": 1},
                                                          "text": 42}]),
            "location not a dict": document_json(sections=[{"source_location": "p1",
                                                            "text": "ok"}]),
            "location value not a number": document_json(
                sections=[{"source_location": {"page": "2"}, "text": "ok"}]),
            "empty location": document_json(sections=[{"source_location": {}, "text": "ok"}]),
            "missing section text": document_json(sections=[{"source_location": {"block": 1}}]),
            "deeply nested": "[" * 100_000 + "]" * 100_000,
        }
        for name, content in cases.items():
            with self.subTest(case=name):
                document_id = self.processed_document()
                if isinstance(content, dict):
                    content = json.dumps(content | {"document_id": document_id})
                self.write_processed_json(document_id, content)

                with self.assertRaises(StorageError) as caught:
                    self.run_chunking(document_id)

                self.assertEqual(caught.exception.safe_message,
                                 processor.CORRUPTED_PROCESSED_MESSAGE)
        self.assertEqual(self.chunk_row_count(), 0)

    def test_processed_json_of_another_document_is_rejected(self):
        document_id = self.processed_document()
        other = json.loads((self.processed_dir / f"{document_id}.json").read_text("utf-8"))
        other["document_id"] = "c" * 32
        self.write_processed_json(document_id, json.dumps(other))

        with self.assertRaises(StorageError) as caught:
            self.run_chunking(document_id)

        self.assertEqual(caught.exception.safe_message, processor.CORRUPTED_PROCESSED_MESSAGE)

    def test_processed_document_without_text_is_rejected(self):
        for sections in [[], [{"source_location": {"block": 1}, "text": "   "}]]:
            with self.subTest(sections=sections):
                document_id = self.processed_document()
                self.write_processed_json(document_id, json.dumps({
                    "schema_version": PROCESSED_SCHEMA_VERSION, "document_id": document_id,
                    "source_filename": "a.txt", "file_type": ".txt", "sections": sections,
                }))

                with self.assertRaises(DocumentContentError) as caught:
                    self.run_chunking(document_id)

                self.assertEqual(caught.exception.safe_message, EMPTY_DOCUMENT_MESSAGE)

    def test_unknown_and_invalid_documents_are_rejected(self):
        database.initialize(self.db_path)

        with self.assertRaises(DocumentNotFoundError):
            self.run_chunking("0" * 32)
        for bad_id in ["../../etc/passwd", "A" * 32, "a" * 32 + ".json", ""]:
            with self.subTest(document_id=bad_id):
                with self.assertRaises(InvalidDocumentIdError):
                    self.run_chunking(bad_id)

    def test_unexpected_crash_gives_generic_message_and_keeps_old_chunks(self):
        document_id = self.processed_document()
        before = self.run_chunking(document_id)
        crash = RuntimeError("secret detail C:\\internal\\path")

        with patch.object(processor, "build_chunks", side_effect=crash):
            with self.assertRaises(ProcessingError) as caught:
                self.run_chunking(document_id)

        self.assertEqual(caught.exception.safe_message, processor.UNEXPECTED_CHUNKING_ERROR_MESSAGE)
        self.assertNotIn("secret", caught.exception.safe_message)
        self.assertEqual(self.stored_chunks(document_id).chunks, before.chunks)

    def test_sqlite_write_failure_is_reported(self):
        document_id = self.processed_document()

        with patch.object(database, "replace_chunks", side_effect=StorageError()):
            with self.assertRaises(StorageError):
                self.run_chunking(document_id)

        self.assertEqual(self.chunk_row_count(), 0)

    def test_reprocessing_during_chunking_is_detected(self):
        document_id = self.processed_document()
        real_build_chunks = processor.build_chunks

        def build_while_document_is_reprocessed(document, config):
            database.claim_for_processing(self.db_path, document_id)  # another request
            return real_build_chunks(document, config)

        with patch.object(processor, "build_chunks", side_effect=build_while_document_is_reprocessed):
            with self.assertRaises(ChunkingConflictError):
                self.run_chunking(document_id)

        self.assertEqual(self.chunk_row_count(), 0)
        self.assertEqual(self.status_of(document_id), DocumentStatus.PROCESSING)


if __name__ == "__main__":
    unittest.main()
