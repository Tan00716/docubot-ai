"""SQLite reads and writes for the "embeddings" table.

The table itself is created with the others in
app/document_processing/database.py, and the same connect() is used, so
foreign keys are enforced and every sqlite3 error becomes a safe StorageError.
All SQL uses "?" placeholders.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from app.document_processing.database import connect, utc_now
from app.document_processing.models import StorageError
from app.embeddings.models import (
    EmbeddingConflictError,
    EmbeddingContract,
    EmbeddingDimensionError,
    EmbeddingRecord,
    NewEmbedding,
    StoredEmbedding,
)
from app.embeddings.vectors import deserialize_vector, serialize_vector, text_sha256

logger = logging.getLogger(__name__)

CORRUPTED_EMBEDDINGS_MESSAGE = "Stored embedding data is corrupted."

RECORD_COLUMNS = (
    "e.chunk_id, e.model_name, e.model_revision, e.embedding_version, e.dimension, "
    "e.max_tokens, e.passage_prefix, e.dtype, e.normalized, e.text_sha256, "
    "e.input_sha256, e.truncated, e.created_at"
)

UPSERT_SQL = """
INSERT INTO embeddings (chunk_id, model_name, model_revision, embedding_version, dimension,
                        max_tokens, passage_prefix, dtype, normalized, text_sha256,
                        input_sha256, truncated, vector, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (chunk_id) DO UPDATE SET
    model_name = excluded.model_name,
    model_revision = excluded.model_revision,
    embedding_version = excluded.embedding_version,
    dimension = excluded.dimension,
    max_tokens = excluded.max_tokens,
    passage_prefix = excluded.passage_prefix,
    dtype = excluded.dtype,
    normalized = excluded.normalized,
    text_sha256 = excluded.text_sha256,
    input_sha256 = excluded.input_sha256,
    truncated = excluded.truncated,
    vector = excluded.vector,
    created_at = excluded.created_at
"""


def _to_record(row: tuple) -> EmbeddingRecord:
    """Turn a database row into an EmbeddingRecord, checking every type."""
    (chunk_id, model_name, revision, version, dimension, max_tokens, prefix, dtype,
     normalized, sha, input_sha, truncated, created) = row
    texts = (chunk_id, model_name, revision, prefix, dtype, sha, input_sha, created)
    numbers = (version, dimension, max_tokens, normalized, truncated)
    if not all(isinstance(value, str) for value in texts):
        raise ValueError("Invalid embedding text field.")
    if not all(isinstance(value, int) for value in numbers):
        raise ValueError("Invalid embedding number field.")
    return EmbeddingRecord(
        chunk_id=chunk_id,
        model_name=model_name,
        model_revision=revision,
        embedding_version=version,
        dimension=dimension,
        max_tokens=max_tokens,
        passage_prefix=prefix,
        dtype=dtype,
        normalized=bool(normalized),
        text_sha256=sha,
        input_sha256=input_sha,
        truncated=bool(truncated),
        created_at=created,
    )


def get_embedding_records(db_path: Path, document_id: str) -> dict[str, EmbeddingRecord]:
    """Metadata of every stored embedding of a document, by chunk_id (no vectors)."""
    with connect(db_path) as connection:
        rows = connection.execute(
            f"SELECT {RECORD_COLUMNS} FROM embeddings AS e "
            "JOIN chunks AS c ON c.chunk_id = e.chunk_id WHERE c.document_id = ?",
            (document_id,),
        ).fetchall()
    try:
        records = [_to_record(row) for row in rows]
    except ValueError as error:
        logger.error("Corrupted embedding metadata for %s: %s", document_id, type(error).__name__)
        raise StorageError(CORRUPTED_EMBEDDINGS_MESSAGE)
    return {record.chunk_id: record for record in records}


def get_embedding(db_path: Path, chunk_id: str) -> StoredEmbedding | None:
    """One stored embedding including its vector, or None if there is none."""
    with connect(db_path) as connection:
        row = connection.execute(
            f"SELECT {RECORD_COLUMNS}, e.vector FROM embeddings AS e WHERE e.chunk_id = ?",
            (chunk_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        record = _to_record(row[:-1])
        vector = deserialize_vector(row[-1], record.dimension)
    except ValueError as error:
        logger.error("Corrupted embedding for chunk %s: %s", chunk_id, type(error).__name__)
        raise StorageError(CORRUPTED_EMBEDDINGS_MESSAGE)
    return StoredEmbedding(record=record, vector=vector)


def save_embeddings(
    db_path: Path, contract: EmbeddingContract, embeddings: Sequence[NewEmbedding]
) -> None:
    """Store (insert or replace) one batch of embeddings in ONE short transaction.

    The model runs BEFORE this is called, so the database is never locked
    while the slow model works. Inside the transaction every chunk is checked
    again: it must still exist and still have exactly the text that was
    embedded. Otherwise nothing from this batch is saved
    (EmbeddingConflictError), so a vector never describes other text.
    """
    if not embeddings:
        return
    chunk_ids = [embedding.chunk_id for embedding in embeddings]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("A chunk may appear only once per batch.")

    now = utc_now()
    try:
        rows = [
            (
                embedding.chunk_id,
                contract.model_name,
                contract.model_revision,
                contract.embedding_version,
                contract.dimension,
                contract.max_tokens,
                contract.passage_prefix,
                contract.dtype,
                int(contract.normalized),
                embedding.text_sha256,
                embedding.input_sha256,
                int(embedding.truncated),
                serialize_vector(embedding.vector, contract.dimension),
                now,
            )
            for embedding in embeddings
        ]
    except ValueError:  # wrong length, NaN or infinity
        raise EmbeddingDimensionError()

    placeholders = ", ".join("?" for _ in chunk_ids)  # only "?" marks, never values
    with connect(db_path) as connection:
        # IMMEDIATE takes the write lock now: nobody can change the chunks
        # between the check below and the write.
        connection.execute("BEGIN IMMEDIATE")
        current_texts = dict(
            connection.execute(
                f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})",
                chunk_ids,
            ).fetchall()
        )
        for embedding in embeddings:
            text = current_texts.get(embedding.chunk_id)
            if text is None or text_sha256(text) != embedding.text_sha256:
                raise EmbeddingConflictError()  # rolls back the whole batch
        connection.executemany(UPSERT_SQL, rows)
