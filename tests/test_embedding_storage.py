"""Tests for storing embeddings in SQLite and for the embedding pipeline.

A small fake model (fake_embeddings.py) is used, so these tests are fast and
offline. Temporary folders and databases only; no .env, no network.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import logging
import sqlite3
import struct
import unittest
from contextlib import closing
from unittest.mock import patch

from app.document_processing import database
from app.document_processing.chunking import ChunkingConfig
from app.document_processing.models import (
    DocumentNotFoundError,
    DocumentNotProcessedError,
    InvalidDocumentIdError,
    ProcessingError,
    StorageError,
)
from app.embeddings import service, storage
from app.embeddings.models import (
    EmbeddingConflictError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingState,
    InvalidEmbeddingConfigError,
    NewEmbedding,
    NoChunksError,
)
from app.embeddings.vectors import serialize_vector, text_sha256, to_float32
from fake_embeddings import FAKE_REVISION, FakeEmbeddingProvider, fake_vector
from test_chunk_storage import ChunkingStorageTestCase
from test_chunking import words

logging.getLogger("app").setLevel(logging.CRITICAL)

MANY_BLOCKS = "\n\n".join(words(40, f"b{n}w") for n in range(30)).encode()  # many chunks


class EmbeddingTestCase(ChunkingStorageTestCase):
    """Adds helpers: a chunked document, running the service, reading rows."""

    def chunked_document(self, data=MANY_BLOCKS, config=None):
        document_id = self.processed_document(data)
        self.run_chunking(document_id, config or ChunkingConfig(300, 0))
        return document_id

    def run_embedding(self, document_id, provider, batch_size=None):
        return service.embed_document(document_id, upload_dir=self.upload_dir,
                                      db_path=self.db_path, provider=provider,
                                      batch_size=batch_size)

    def status(self, document_id, provider):
        return service.get_embedding_status(document_id, upload_dir=self.upload_dir,
                                            db_path=self.db_path, contract=provider.contract)

    def chunks(self, document_id):
        return self.stored_chunks(document_id).chunks

    def embedding_row_count(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def execute(self, sql, parameters=()):
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute(sql, parameters)


class EmbeddingStorageTests(EmbeddingTestCase):
    def test_initialize_creates_embeddings_table(self):
        database.initialize(self.db_path)

        with closing(sqlite3.connect(self.db_path)) as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        self.assertIn(("embeddings",), tables)

    def test_embeddings_are_stored_and_read_back(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()

        self.run_embedding(document_id, provider)

        chunk = self.chunks(document_id)[0]
        stored = storage.get_embedding(self.db_path, chunk.chunk_id)
        self.assertEqual(stored.vector, provider.passage_vector(chunk.text))
        self.assertNotEqual(stored.vector, provider.embed_query(chunk.text),
                            "a stored chunk is a passage, not a query")
        self.assertEqual(stored.record.model_name, provider.contract.model_name)
        self.assertEqual(stored.record.model_revision, FAKE_REVISION)
        self.assertEqual(stored.record.embedding_version, 2)
        self.assertEqual(stored.record.dimension, 8)
        self.assertEqual(stored.record.max_tokens, 512)
        self.assertEqual(stored.record.passage_prefix, "passage: ")
        self.assertEqual(stored.record.dtype, "float32")
        self.assertTrue(stored.record.normalized)
        self.assertIsInstance(stored.record.created_at, str)

    def test_text_hash_is_of_the_original_text_and_input_hash_of_the_model_input(self):
        document_id = self.chunked_document()
        self.run_embedding(document_id, FakeEmbeddingProvider())

        chunk = self.chunks(document_id)[0]
        record = storage.get_embedding(self.db_path, chunk.chunk_id).record
        self.assertEqual(record.text_sha256, text_sha256(chunk.text))
        self.assertEqual(record.input_sha256, text_sha256("passage: " + chunk.text))
        self.assertNotEqual(record.text_sha256, record.input_sha256)

    def test_prefix_is_never_saved_as_chunk_text(self):
        document_id = self.chunked_document()
        original = [c.text for c in self.chunks(document_id)]
        provider = FakeEmbeddingProvider()

        self.run_embedding(document_id, provider)

        self.assertEqual([c.text for c in self.chunks(document_id)], original)
        self.assertFalse(any(c.text.startswith("passage:") for c in self.chunks(document_id)))
        self.assertEqual(provider.model_inputs, ["passage: " + text for text in original])

    def test_vector_is_stored_as_compact_float32_blob(self):
        document_id = self.chunked_document()
        self.run_embedding(document_id, FakeEmbeddingProvider(dimension=8))

        with closing(sqlite3.connect(self.db_path)) as connection:
            kind, size = connection.execute(
                "SELECT typeof(vector), length(vector) FROM embeddings LIMIT 1").fetchone()

        self.assertEqual((kind, size), ("blob", 8 * 4))

    def test_unknown_chunk_has_no_embedding(self):
        database.initialize(self.db_path)

        self.assertIsNone(storage.get_embedding(self.db_path, "missing-chunk"))

    def test_foreign_key_rejects_embedding_of_unknown_chunk(self):
        database.initialize(self.db_path)

        with self.assertRaises(StorageError):
            with database.connect(self.db_path) as connection:
                connection.execute(
                    "INSERT INTO embeddings (chunk_id, model_name, embedding_version, dimension,"
                    " dtype, normalized, text_sha256, truncated, vector, created_at)"
                    " VALUES ('no-such-chunk', 'm', 1, 1, 'float32', 1, ?, 0, ?, 't')",
                    ("0" * 64, serialize_vector([1.0], 1)))

    def test_database_rejects_a_blob_of_the_wrong_size(self):
        document_id = self.chunked_document()
        self.run_embedding(document_id, FakeEmbeddingProvider())

        with self.assertRaises(sqlite3.IntegrityError):
            self.execute("UPDATE embeddings SET vector = ?", (b"\x00" * 5,))

    def test_one_embedding_per_chunk_is_enforced(self):
        document_id = self.chunked_document()
        self.run_embedding(document_id, FakeEmbeddingProvider())
        chunk_id = self.chunks(document_id)[0].chunk_id

        with self.assertRaises(sqlite3.IntegrityError):
            self.execute("INSERT INTO embeddings SELECT * FROM embeddings WHERE chunk_id = ?",
                         (chunk_id,))

    def test_saving_for_a_missing_chunk_is_rejected(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        ghost = NewEmbedding("no-such-chunk", "0" * 64, "0" * 64, provider.embed_query("x"),
                             False)

        with self.assertRaises(EmbeddingConflictError):
            storage.save_embeddings(self.db_path, provider.contract, [ghost])

        self.assertEqual(self.embedding_row_count(), 0)

    def test_saving_a_vector_for_changed_text_is_rejected(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        chunk = self.chunks(document_id)[0]
        outdated = NewEmbedding(chunk.chunk_id, text_sha256("older text"),
                                text_sha256("passage: older text"),
                                provider.passage_vector("older text"), False)

        with self.assertRaises(EmbeddingConflictError):
            storage.save_embeddings(self.db_path, provider.contract, [outdated])

    def test_wrong_dimension_is_rejected_before_writing(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider(dimension=8)
        chunk = self.chunks(document_id)[0]
        short = NewEmbedding(chunk.chunk_id, text_sha256(chunk.text),
                             text_sha256("passage: " + chunk.text), (0.5, 0.5), False)

        with self.assertRaises(EmbeddingDimensionError):
            storage.save_embeddings(self.db_path, provider.contract, [short])

        self.assertEqual(self.embedding_row_count(), 0)

    def test_failing_batch_writes_nothing(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        good, bad = self.chunks(document_id)[:2]
        batch = [NewEmbedding(good.chunk_id, text_sha256(good.text),
                              text_sha256("passage: " + good.text),
                              provider.passage_vector(good.text), False),
                 NewEmbedding(bad.chunk_id, text_sha256("different"),
                              text_sha256("passage: different"),
                              provider.passage_vector("different"), False)]

        with self.assertRaises(EmbeddingConflictError):
            storage.save_embeddings(self.db_path, provider.contract, batch)

        self.assertEqual(self.embedding_row_count(), 0, "no half-written batch")

    def test_corrupted_stored_embeddings_are_reported_safely(self):
        nan_blob = struct.pack("<8f", float("nan"), *[0.0] * 7)
        read_one = lambda document_id, chunk_id: storage.get_embedding(self.db_path, chunk_id)
        read_all = lambda document_id, chunk_id: storage.get_embedding_records(self.db_path,
                                                                               document_id)
        corruptions = [
            ("UPDATE embeddings SET vector = ?", (nan_blob,), read_one),
            ("UPDATE embeddings SET model_name = CAST('m' AS BLOB)", (), read_one),
            ("UPDATE embeddings SET model_name = CAST('m' AS BLOB)", (), read_all),
            ("UPDATE embeddings SET dimension = 'x'", (), read_all),
            ("UPDATE embeddings SET max_tokens = 'x'", (), read_all),
            ("UPDATE embeddings SET passage_prefix = CAST('p' AS BLOB)", (), read_all),
            ("UPDATE embeddings SET input_sha256 = CAST('h' AS BLOB)", (), read_one),
        ]
        for sql, parameters, read in corruptions:
            with self.subTest(sql=sql, reader=read):
                document_id = self.chunked_document()
                self.run_embedding(document_id, FakeEmbeddingProvider())
                chunk_id = self.chunks(document_id)[0].chunk_id
                with closing(sqlite3.connect(self.db_path)) as connection, connection:
                    connection.execute("PRAGMA ignore_check_constraints = ON")
                    connection.execute(sql + " WHERE chunk_id = ?", (*parameters, chunk_id))

                with self.assertRaises(StorageError) as caught:
                    read(document_id, chunk_id)

                self.assertEqual(caught.exception.safe_message,
                                 storage.CORRUPTED_EMBEDDINGS_MESSAGE)

    def test_rechunking_deletes_the_old_embeddings(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        self.run_embedding(document_id, provider)

        self.run_chunking(document_id, ChunkingConfig(500, 0))

        self.assertEqual(self.embedding_row_count(), 0)
        self.assertEqual(self.status(document_id, provider).embedded_chunks, 0)

    def test_reprocessing_deletes_the_old_embeddings(self):
        document_id = self.chunked_document()
        self.run_embedding(document_id, FakeEmbeddingProvider())

        self.run_processing(document_id)

        self.assertEqual(self.embedding_row_count(), 0)


class EmbedDocumentTests(EmbeddingTestCase):
    def test_first_run_embeds_every_chunk(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()

        summary = self.run_embedding(document_id, provider)

        total = len(self.chunks(document_id))
        self.assertGreater(total, 5)
        self.assertEqual((summary.total_chunks, summary.embedded_count, summary.skipped_count,
                          summary.stale_reembedded_count), (total, total, 0, 0))
        self.assertEqual(self.embedding_row_count(), total)
        self.assertEqual(summary.contract, provider.contract)

    def test_second_run_embeds_nothing(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        self.run_embedding(document_id, provider)
        provider.batch_calls.clear()

        summary = self.run_embedding(document_id, provider)

        total = len(self.chunks(document_id))
        self.assertEqual((summary.embedded_count, summary.skipped_count), (0, total))
        self.assertEqual(provider.batch_calls, [], "the model must not be called again")
        self.assertEqual(self.embedding_row_count(), total)

    def test_chunks_are_sent_in_batches_in_chunk_order(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        total = len(self.chunks(document_id))

        self.run_embedding(document_id, provider, batch_size=4)

        self.assertEqual(sum(provider.batch_calls), total)
        self.assertTrue(all(size <= 4 for size in provider.batch_calls))
        self.assertEqual(len(provider.batch_calls), -(-total // 4))  # ceil(total / 4)
        self.assertEqual(provider.embedded_texts, [c.text for c in self.chunks(document_id)])

    def test_default_batch_size_comes_from_the_provider_config(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider(batch_size=3)

        self.run_embedding(document_id, provider)

        self.assertEqual(max(provider.batch_calls), 3)

    def test_invalid_batch_size_is_rejected_before_any_work(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        for bad in [0, -1, 65, 2.5]:
            with self.subTest(batch_size=bad):
                with self.assertRaises(InvalidEmbeddingConfigError):
                    self.run_embedding(document_id, provider, batch_size=bad)
        self.assertEqual(provider.batch_calls, [])

    def test_changed_chunk_text_is_embedded_again(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        self.run_embedding(document_id, provider)
        chunk = self.chunks(document_id)[2]
        # The text changes while the chunk_id stays the same.
        self.execute("UPDATE chunks SET text = ?, char_count = ? WHERE chunk_id = ?",
                     ("Brand new text.", len("Brand new text."), chunk.chunk_id))
        self.assertEqual(self.status(document_id, provider).stale_embeddings, 1)
        provider.embedded_texts.clear()

        summary = self.run_embedding(document_id, provider)

        self.assertEqual((summary.embedded_count, summary.stale_reembedded_count), (1, 1))
        self.assertEqual(provider.embedded_texts, ["Brand new text."])
        stored = storage.get_embedding(self.db_path, chunk.chunk_id)
        self.assertEqual(stored.record.text_sha256, text_sha256("Brand new text."))
        self.assertEqual(stored.record.input_sha256, text_sha256("passage: Brand new text."))
        self.assertEqual(stored.vector, provider.passage_vector("Brand new text."))

    def test_new_model_replaces_old_vectors_without_duplicates(self):
        document_id = self.chunked_document()
        old = FakeEmbeddingProvider(model_name="fake/old-model")
        self.run_embedding(document_id, old)
        new = FakeEmbeddingProvider(model_name="fake/new-model")
        total = len(self.chunks(document_id))
        self.assertEqual(self.status(document_id, new).stale_embeddings, total)

        summary = self.run_embedding(document_id, new)

        self.assertEqual(summary.stale_reembedded_count, total)
        self.assertEqual(self.embedding_row_count(), total, "one active embedding per chunk")
        records = storage.get_embedding_records(self.db_path, document_id)
        self.assertEqual({r.model_name for r in records.values()}, {"fake/new-model"})

    def test_new_embedding_version_or_normalization_replaces_vectors(self):
        document_id = self.chunked_document()
        self.run_embedding(document_id, FakeEmbeddingProvider())
        total = len(self.chunks(document_id))

        for provider in [FakeEmbeddingProvider(embedding_version=3),
                         FakeEmbeddingProvider(embedding_version=3, normalize=False),
                         FakeEmbeddingProvider(embedding_version=3, normalize=False,
                                               revision="e" * 40),
                         FakeEmbeddingProvider(embedding_version=3, normalize=False,
                                               revision="e" * 40, max_tokens=256)]:
            with self.subTest(contract=provider.contract):
                summary = self.run_embedding(document_id, provider)

                self.assertEqual(summary.stale_reembedded_count, total)
                self.assertEqual(self.status(document_id, provider).state, EmbeddingState.COMPLETE)

    def test_new_dimension_replaces_vectors(self):
        document_id = self.chunked_document()
        self.run_embedding(document_id, FakeEmbeddingProvider(dimension=8))

        wider = FakeEmbeddingProvider(dimension=16)
        self.run_embedding(document_id, wider)

        chunk = self.chunks(document_id)[0]
        self.assertEqual(len(storage.get_embedding(self.db_path, chunk.chunk_id).vector), 16)

    def test_duplicate_chunk_texts_each_get_their_own_embedding(self):
        document_id = self.chunked_document(b"Same text.\n\nSame text.", ChunkingConfig(10, 0))
        provider = FakeEmbeddingProvider()

        self.run_embedding(document_id, provider)

        first, second = self.chunks(document_id)
        self.assertEqual(first.text, second.text)
        self.assertEqual(storage.get_embedding(self.db_path, first.chunk_id).vector,
                         storage.get_embedding(self.db_path, second.chunk_id).vector)
        self.assertEqual(self.embedding_row_count(), 2)

    def test_interrupted_run_keeps_finished_batches_and_resumes(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        real_embed = provider.embed_documents
        calls = {"count": 0}

        def fail_on_third_batch(texts):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("simulated crash")
            return real_embed(texts)

        with patch.object(provider, "embed_documents", side_effect=fail_on_third_batch):
            with self.assertRaises(ProcessingError) as caught:
                self.run_embedding(document_id, provider, batch_size=2)
        self.assertEqual(caught.exception.safe_message, service.UNEXPECTED_EMBEDDING_ERROR_MESSAGE)
        self.assertEqual(self.embedding_row_count(), 4, "two finished batches of 2")

        summary = self.run_embedding(document_id, provider, batch_size=2)

        self.assertEqual(summary.skipped_count, 4)
        self.assertEqual(self.embedding_row_count(), len(self.chunks(document_id)))

    def test_rechunking_during_embedding_is_detected(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()
        real_embed = provider.embed_documents

        def embed_while_rechunked(texts):
            self.run_chunking(document_id, ChunkingConfig(600, 0))  # another request
            return real_embed(texts)

        with patch.object(provider, "embed_documents", side_effect=embed_while_rechunked):
            with self.assertRaises(EmbeddingConflictError):
                self.run_embedding(document_id, provider)

        self.assertEqual(self.embedding_row_count(), 0)

    def test_document_must_exist_be_processed_and_chunked(self):
        provider = FakeEmbeddingProvider()
        database.initialize(self.db_path)
        with self.assertRaises(DocumentNotFoundError):
            self.run_embedding("0" * 32, provider)
        with self.assertRaises(InvalidDocumentIdError):
            self.run_embedding("../../etc", provider)
        with self.assertRaises(DocumentNotProcessedError):
            self.run_embedding(self.add_document(), provider)
        with self.assertRaises(NoChunksError):
            self.run_embedding(self.processed_document(), provider)
        self.assertEqual(provider.batch_calls, [])

    def test_provider_that_skips_the_passage_prefix_is_rejected(self):
        document_id = self.chunked_document()
        for wrong_prefix in ["", "query: ", "Passage: "]:
            with self.subTest(prefix=wrong_prefix):
                provider = FakeEmbeddingProvider(wrong_passage_prefix=wrong_prefix)

                with self.assertRaises(EmbeddingError):
                    self.run_embedding(document_id, provider)

                self.assertEqual(self.embedding_row_count(), 0, "nothing is saved")

    def test_provider_returning_too_few_vectors_is_rejected(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider()

        with patch.object(provider, "embed_documents", return_value=[]):
            with self.assertRaises(ProcessingError):
                self.run_embedding(document_id, provider)

        self.assertEqual(self.embedding_row_count(), 0)

    def test_database_failure_is_reported(self):
        document_id = self.chunked_document()

        with patch.object(storage, "save_embeddings", side_effect=StorageError()):
            with self.assertRaises(StorageError):
                self.run_embedding(document_id, FakeEmbeddingProvider())


# The "embeddings" table exactly as Batch 5 (commit b1fd9b5) created it.
BATCH5_EMBEDDINGS_SCHEMA = """
CREATE TABLE embeddings (
    chunk_id          TEXT PRIMARY KEY
                      REFERENCES chunks (chunk_id) ON DELETE CASCADE,
    model_name        TEXT NOT NULL,
    embedding_version INTEGER NOT NULL,
    dimension         INTEGER NOT NULL CHECK (dimension > 0),
    dtype             TEXT NOT NULL CHECK (dtype = 'float32'),
    normalized        INTEGER NOT NULL CHECK (normalized IN (0, 1)),
    text_sha256       TEXT NOT NULL CHECK (length(text_sha256) = 64),
    truncated         INTEGER NOT NULL CHECK (truncated IN (0, 1)),
    vector            BLOB NOT NULL
                      CHECK (typeof(vector) = 'blob' AND length(vector) = dimension * 4),
    created_at        TEXT NOT NULL
)
"""
OLD_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class OldModelUpgradeTests(EmbeddingTestCase):
    """A database from Batch 5 holds vectors of the old model (contract v1)."""

    def make_batch5_database(self):
        """A chunked document whose chunks all have Batch 5 (old model) vectors."""
        document_id = self.chunked_document()
        chunks = self.chunks(document_id)
        old_vector = serialize_vector(to_float32([1 / 384 ** 0.5] * 384), 384)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute("DROP TABLE embeddings")
            connection.execute(BATCH5_EMBEDDINGS_SCHEMA)
            for chunk in chunks:
                connection.execute(
                    "INSERT INTO embeddings VALUES (?, ?, 1, 384, 'float32', 1, ?, 1, ?, 't')",
                    (chunk.chunk_id, OLD_MODEL, text_sha256(chunk.text), old_vector))
        return document_id

    def columns(self):
        with closing(sqlite3.connect(self.db_path)) as connection:
            return {row[1] for row in connection.execute("PRAGMA table_info(embeddings)")}

    def test_batch5_table_gets_the_new_columns_and_keeps_chunks(self):
        document_id = self.make_batch5_database()
        chunks_before = self.chunks(document_id)  # this connect() runs the upgrade
        fresh_path = self.db_path.with_name("fresh.sqlite3")
        database.initialize(fresh_path)
        with closing(sqlite3.connect(fresh_path)) as connection:
            fresh_columns = {row[1] for row in connection.execute("PRAGMA table_info(embeddings)")}

        self.assertEqual(self.columns(), fresh_columns)
        self.assertTrue(set(database.EMBEDDING_CONTRACT_COLUMNS) <= self.columns())
        self.assertEqual(self.chunks(document_id), chunks_before, "chunk IDs and text unchanged")
        self.assertEqual(self.embedding_row_count(), len(chunks_before), "nothing deleted")

    def test_upgrade_is_safe_to_repeat(self):
        self.make_batch5_database()

        for _ in range(3):
            database.initialize(self.db_path)

        self.assertTrue(set(database.EMBEDDING_CONTRACT_COLUMNS) <= self.columns())

    def test_old_model_vectors_are_stale_and_never_counted_as_valid(self):
        document_id = self.make_batch5_database()
        provider = FakeEmbeddingProvider()

        result = self.status(document_id, provider)

        total = len(self.chunks(document_id))
        self.assertEqual((result.embedded_chunks, result.stale_embeddings,
                          result.missing_embeddings, result.truncated_chunks), (0, total, 0, 0))
        self.assertEqual(result.state, EmbeddingState.INCOMPLETE)

    def test_embedding_again_replaces_every_old_vector(self):
        document_id = self.make_batch5_database()
        provider = FakeEmbeddingProvider()
        chunk_ids = [c.chunk_id for c in self.chunks(document_id)]

        summary = self.run_embedding(document_id, provider)

        self.assertEqual(summary.stale_reembedded_count, len(chunk_ids))
        self.assertEqual(self.status(document_id, provider).state, EmbeddingState.COMPLETE)
        records = storage.get_embedding_records(self.db_path, document_id)
        self.assertEqual(sorted(records), sorted(chunk_ids), "same chunk IDs, one row each")
        self.assertEqual({(r.model_name, r.embedding_version, r.passage_prefix)
                          for r in records.values()},
                         {(provider.contract.model_name, 2, "passage: ")})
        self.assertEqual([c.chunk_id for c in self.chunks(document_id)], chunk_ids)

    def test_half_migrated_document_never_mixes_the_two_models(self):
        document_id = self.chunked_document()
        old = FakeEmbeddingProvider(model_name="fake/old-model", embedding_version=1)
        new = FakeEmbeddingProvider()
        self.run_embedding(document_id, old)
        real_embed = new.embed_documents
        calls = {"count": 0}

        def stop_after_first_batch(texts):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("simulated crash")
            return real_embed(texts)

        with patch.object(new, "embed_documents", side_effect=stop_after_first_batch):
            with self.assertRaises(ProcessingError):
                self.run_embedding(document_id, new, batch_size=3)

        total = len(self.chunks(document_id))
        new_status, old_status = self.status(document_id, new), self.status(document_id, old)
        self.assertEqual((new_status.embedded_chunks, new_status.stale_embeddings), (3, total - 3))
        self.assertEqual((old_status.embedded_chunks, old_status.stale_embeddings), (total - 3, 3))
        self.assertEqual(new_status.state, EmbeddingState.INCOMPLETE)


class EmbeddingStatusTests(EmbeddingTestCase):
    def test_status_counts_missing_valid_stale_and_truncated(self):
        document_id = self.chunked_document()
        provider = FakeEmbeddingProvider(truncate_after=200)
        total = len(self.chunks(document_id))
        self.assertEqual(self.status(document_id, provider).missing_embeddings, total)

        self.run_embedding(document_id, provider)
        first = self.chunks(document_id)[0]
        self.execute("UPDATE embeddings SET text_sha256 = ? WHERE chunk_id = ?",
                     ("f" * 64, first.chunk_id))

        result = self.status(document_id, provider)
        long_chunks = sum(1 for c in self.chunks(document_id)[1:] if len(c.text) > 200)
        self.assertEqual((result.total_chunks, result.embedded_chunks, result.missing_embeddings,
                          result.stale_embeddings), (total, total - 1, 0, 1))
        self.assertEqual(result.truncated_chunks, long_chunks)
        self.assertGreater(long_chunks, 0)
        self.assertEqual(result.state, EmbeddingState.INCOMPLETE)

    def test_status_is_complete_after_embedding_and_no_chunks_before_chunking(self):
        provider = FakeEmbeddingProvider()
        unchunked = self.processed_document()
        self.assertEqual(self.status(unchunked, provider).state, EmbeddingState.NO_CHUNKS)

        document_id = self.chunked_document()
        self.run_embedding(document_id, provider)

        self.assertEqual(self.status(document_id, provider).state, EmbeddingState.COMPLETE)

    def test_fake_vectors_are_deterministic(self):
        self.assertEqual(fake_vector("a", 8), fake_vector("a", 8))
        self.assertEqual(to_float32(fake_vector("a", 8)), to_float32(fake_vector("a", 8)))


if __name__ == "__main__":
    unittest.main()
