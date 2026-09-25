"""SQLite storage for document metadata, processing status and chunks.

SQLite is a small database that lives in one local file and ships with
Python (no server, no extra package). Every function opens a short-lived
connection, so it is safe to call from FastAPI's worker threads.

Tables:
    documents  - one row per uploaded document (processing status lives here)
    chunks     - the chunks of each document; a document is "chunked" when
                 it has rows here (chunking status is never stored separately)
    embeddings - one vector per chunk (queries live in app/embeddings/storage.py)

All SQL uses "?" placeholders. Values are never pasted into SQL strings.
Any sqlite3 error is turned into StorageError (with a safe message).
"""

import json
import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.document_processing.models import (
    Chunk,
    ChunkingConflictError,
    ChunkSet,
    DocumentRecord,
    DocumentStatus,
    StorageError,
    parse_source_location,
)

logger = logging.getLogger(__name__)

# A crash (e.g. the server is killed) could leave a row in "processing".
# After this long, the document may be processed again instead of being stuck.
STALE_PROCESSING_AFTER = timedelta(minutes=10)

_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in DocumentStatus)

DOCUMENTS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    original_filename TEXT NOT NULL,
    stored_filename   TEXT NOT NULL,
    extension         TEXT NOT NULL,
    content_type      TEXT,
    size_bytes        INTEGER NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ({_STATUS_VALUES})),
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    processed_path    TEXT,
    error_message     TEXT
)
"""

# source_locations is a JSON list, e.g. [{"page": 2, "char_start": 0, "char_end": 812}].
# The file name and type are not copied here; they are read from "documents".
# UNIQUE(document_id, chunk_index) makes duplicate chunks impossible.
CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL
                     REFERENCES documents (document_id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL CHECK (chunk_index >= 0),
    text             TEXT NOT NULL CHECK (length(text) > 0),
    char_count       INTEGER NOT NULL CHECK (char_count > 0),
    source_locations TEXT NOT NULL,
    chunking_version INTEGER NOT NULL,
    chunk_size       INTEGER NOT NULL,
    chunk_overlap    INTEGER NOT NULL,
    created_at       TEXT NOT NULL,
    UNIQUE (document_id, chunk_index)
)
"""

# Embedding-contract columns added in Batch 5A (embedding version 2). A
# database created by Batch 5 gets them with ALTER TABLE (see
# _add_missing_embedding_columns). Its old rows keep these defaults, which
# never match a real contract, so they are reported as stale and replaced
# the next time the document is embedded. Chunk text is never touched.
EMBEDDING_CONTRACT_COLUMNS = {
    "model_revision": "TEXT NOT NULL DEFAULT ''",  # Hugging Face commit of the model files
    "max_tokens": "INTEGER NOT NULL DEFAULT 0",  # the model's token limit
    "passage_prefix": "TEXT NOT NULL DEFAULT ''",  # added before the chunk text, e.g. "passage: "
    "input_sha256": "TEXT NOT NULL DEFAULT ''",  # SHA-256 of passage_prefix + chunk text
}

_CONTRACT_COLUMNS_SQL = "".join(
    f",\n    {name} {definition}" for name, definition in EMBEDDING_CONTRACT_COLUMNS.items()
)

# One active embedding per chunk (chunk_id is the primary key). vector is a
# float32 BLOB: dimension * 4 bytes. Deleting a chunk (re-chunking or
# re-processing) deletes its embedding too (ON DELETE CASCADE), so a vector
# can never outlive the text it was made from. text_sha256 is the hash of the
# ORIGINAL chunk text (no prefix). Queries: app/embeddings/storage.py.
EMBEDDINGS_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS embeddings (
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
    created_at        TEXT NOT NULL{_CONTRACT_COLUMNS_SQL}
)
"""

SCHEMA_STATEMENTS = (DOCUMENTS_SCHEMA, CHUNKS_SCHEMA, EMBEDDINGS_SCHEMA)

COLUMNS = (
    "document_id, original_filename, stored_filename, extension, content_type, "
    "size_bytes, status, created_at, updated_at, processed_path, error_message"
)

CHUNK_INSERT_COLUMNS = (
    "chunk_id, document_id, chunk_index, text, char_count, source_locations, "
    "chunking_version, chunk_size, chunk_overlap, created_at"
)

# The columns that to_chunk() expects, in this order. Every query that
# reads chunks (here and in app/search/storage.py) selects exactly these.
CHUNK_SELECT_COLUMNS = (
    "c.chunk_id, c.document_id, c.chunk_index, c.text, c.char_count, "
    "c.source_locations, c.chunking_version, c.chunk_size, c.chunk_overlap, "
    "c.created_at, d.original_filename, d.extension"
)
CHUNK_SELECT_COLUMN_COUNT = len(CHUNK_SELECT_COLUMNS.split(","))  # 12

CORRUPTED_CHUNKS_MESSAGE = "Stored chunk data is corrupted."


def _add_missing_embedding_columns(connection: sqlite3.Connection) -> None:
    """Upgrade an older "embeddings" table in place (safe to run every time).

    Column names and definitions come only from EMBEDDING_CONTRACT_COLUMNS
    above, never from outside input, so building this SQL is safe.
    """
    existing = {row[1] for row in connection.execute("PRAGMA table_info(embeddings)")}
    for name, definition in EMBEDDING_CONTRACT_COLUMNS.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE embeddings ADD COLUMN {name} {definition}")
            logger.info("Added column embeddings.%s", name)


def utc_now() -> str:
    """Current time as text, e.g. "2026-09-24T01:30:00.123456+00:00"."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database (creating file and tables if needed) in a transaction.

    The transaction is committed when the "with" block ends normally and
    rolled back if an error happens inside it.
    """
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as connection:
            # SQLite only enforces REFERENCES (foreign keys) when this is
            # switched on, and it must be switched on for every connection.
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                for statement in SCHEMA_STATEMENTS:
                    connection.execute(statement)
                _add_missing_embedding_columns(connection)
                yield connection
    except sqlite3.Error as error:
        logger.error("SQLite error: %s", type(error).__name__)
        raise StorageError("Could not access the document database.")
    except OSError as error:
        logger.error("Could not open the database folder: %s", type(error).__name__)
        raise StorageError("Could not access the document database.")


def initialize(db_path: Path) -> None:
    """Create the database file and its tables if they do not exist."""
    with connect(db_path):
        pass


def _to_record(row: tuple) -> DocumentRecord:
    values = list(row)
    values[6] = DocumentStatus(values[6])
    return DocumentRecord(*values)


def create_document(
    db_path: Path,
    *,
    document_id: str,
    original_filename: str,
    stored_filename: str,
    extension: str,
    content_type: str | None,
    size_bytes: int,
) -> DocumentRecord:
    """Insert a new row with status "uploaded".

    If a row with this document_id already exists, it is left unchanged.
    """
    now = utc_now()
    with connect(db_path) as connection:
        connection.execute(
            f"INSERT OR IGNORE INTO documents ({COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
            (
                document_id,
                original_filename,
                stored_filename,
                extension,
                content_type,
                size_bytes,
                DocumentStatus.UPLOADED.value,
                now,
                now,
            ),
        )
    record = get_document(db_path, document_id)
    if record is None:  # should never happen: the row was just inserted
        raise StorageError("Could not save the document record.")
    return record


def get_document(db_path: Path, document_id: str) -> DocumentRecord | None:
    """Return the row for this document_id, or None if it does not exist."""
    with connect(db_path) as connection:
        row = connection.execute(
            f"SELECT {COLUMNS} FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
    return _to_record(row) if row else None


def claim_for_processing(db_path: Path, document_id: str) -> bool:
    """Move a document to "processing".

    Returns False if another request is already processing it (and that
    processing is not stale). The check and the update happen in one SQL
    statement, so two requests cannot both claim the same document.

    Chunks are made from the processed output, so processing a document again
    deletes its old chunks in the same transaction. Chunks can therefore never
    belong to an older version of the processed output.
    """
    now = datetime.now(UTC)
    stale_before = (now - STALE_PROCESSING_AFTER).isoformat(timespec="microseconds")
    with connect(db_path) as connection:
        cursor = connection.execute(
            "UPDATE documents "
            "SET status = ?, updated_at = ?, processed_path = NULL, error_message = NULL "
            "WHERE document_id = ? AND (status != ? OR updated_at < ?)",
            (
                DocumentStatus.PROCESSING.value,
                utc_now(),
                document_id,
                DocumentStatus.PROCESSING.value,
                stale_before,
            ),
        )
        claimed = cursor.rowcount == 1
        if claimed:
            connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        return claimed


def mark_completed(db_path: Path, document_id: str, processed_path: str) -> None:
    """processing -> completed, remembering where the processed JSON is."""
    _finish(db_path, document_id, DocumentStatus.COMPLETED, processed_path, None)


def mark_failed(db_path: Path, document_id: str, error_message: str) -> None:
    """processing -> failed, storing a short, safe error message."""
    _finish(db_path, document_id, DocumentStatus.FAILED, None, error_message)


def _finish(
    db_path: Path,
    document_id: str,
    status: DocumentStatus,
    processed_path: str | None,
    error_message: str | None,
) -> None:
    with connect(db_path) as connection:
        connection.execute(
            "UPDATE documents "
            "SET status = ?, updated_at = ?, processed_path = ?, error_message = ? "
            "WHERE document_id = ? AND status = ?",
            (
                status.value,
                utc_now(),
                processed_path,
                error_message,
                document_id,
                DocumentStatus.PROCESSING.value,
            ),
        )


# --- Chunks -------------------------------------------------------------------


def replace_chunks(
    db_path: Path,
    document_id: str,
    chunks: Sequence[Chunk],
    *,
    expected_updated_at: str,
) -> str:
    """Store `chunks` as the complete chunk set of one document.

    The document's old chunks are deleted and the new ones inserted in ONE
    transaction: either everything is saved or, on any error, nothing changes.
    Running it again with the same chunks produces the same rows, so chunks
    are never duplicated (idempotent).

    expected_updated_at is the document's updated_at at the moment its
    processed output was loaded. If the document was re-processed since then,
    the chunks may be outdated: ChunkingConflictError is raised, nothing saved.

    Returns the created_at timestamp stored with the chunks.
    """
    if any(chunk.document_id != document_id for chunk in chunks):
        raise ValueError("Every chunk must belong to the document being replaced.")

    now = utc_now()
    rows = [
        (
            chunk.chunk_id,
            chunk.document_id,
            chunk.chunk_index,
            chunk.text,
            chunk.char_count,
            json.dumps(list(chunk.source_locations), ensure_ascii=False),
            chunk.chunking_version,
            chunk.chunk_size,
            chunk.chunk_overlap,
            now,
        )
        for chunk in chunks
    ]
    with connect(db_path) as connection:
        # The DELETE starts the write transaction. Until it is committed, no
        # other request can change this document (e.g. start re-processing it).
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        row = connection.execute(
            "SELECT status, updated_at FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row != (DocumentStatus.COMPLETED.value, expected_updated_at):
            raise ChunkingConflictError()  # rolls back the DELETE as well
        connection.executemany(
            f"INSERT INTO chunks ({CHUNK_INSERT_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return now


def get_chunk_set(db_path: Path, document_id: str) -> ChunkSet:
    """Return all chunks of a document in chunk_index order (maybe none)."""
    with connect(db_path) as connection:
        rows = connection.execute(
            f"SELECT {CHUNK_SELECT_COLUMNS} "
            "FROM chunks AS c JOIN documents AS d ON d.document_id = c.document_id "
            "WHERE c.document_id = ? ORDER BY c.chunk_index",
            (document_id,),
        ).fetchall()

    try:
        chunks = tuple(to_chunk(row) for row in rows)
    except (ValueError, TypeError) as error:
        logger.error("Corrupted chunk data for document %s: %s", document_id, type(error).__name__)
        raise StorageError(CORRUPTED_CHUNKS_MESSAGE)

    # Stored chunks must be numbered 0, 1, 2, ... without gaps.
    if [chunk.chunk_index for chunk in chunks] != list(range(len(chunks))):
        logger.error("Chunk indexes of document %s are not continuous", document_id)
        raise StorageError(CORRUPTED_CHUNKS_MESSAGE)

    created_at = rows[0][9] if rows else None
    return ChunkSet(document_id=document_id, chunks=chunks, created_at=created_at)


def to_chunk(row: tuple) -> Chunk:
    """Turn one database row (CHUNK_SELECT_COLUMNS) into a Chunk, checking the stored JSON.

    Raises ValueError or TypeError for corrupted rows.
    """
    text, char_count = row[3], row[4]
    if not isinstance(text, str) or char_count != len(text):
        raise ValueError("Invalid chunk text.")

    locations = json.loads(row[5])
    if not isinstance(locations, list) or not locations:
        raise ValueError("Invalid chunk source locations.")

    return Chunk(
        chunk_id=row[0],
        document_id=row[1],
        chunk_index=row[2],
        text=text,
        char_count=char_count,
        source_filename=row[10],
        file_type=row[11],
        source_locations=tuple(parse_source_location(item) for item in locations),
        chunking_version=row[6],
        chunk_size=row[7],
        chunk_overlap=row[8],
    )
