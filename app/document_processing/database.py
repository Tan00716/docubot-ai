"""SQLite storage for document metadata and processing status.

SQLite is a small database that lives in one local file and ships with
Python (no server, no extra package). Every function opens a short-lived
connection, so it is safe to call from FastAPI's worker threads.

All SQL uses "?" placeholders. Values are never pasted into SQL strings.
Any sqlite3 error is turned into StorageError (with a safe message).
"""

import logging
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.document_processing.models import (
    DocumentRecord,
    DocumentStatus,
    StorageError,
)

logger = logging.getLogger(__name__)

# A crash (e.g. the server is killed) could leave a row in "processing".
# After this long, the document may be processed again instead of being stuck.
STALE_PROCESSING_AFTER = timedelta(minutes=10)

_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in DocumentStatus)

SCHEMA = f"""
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

COLUMNS = (
    "document_id, original_filename, stored_filename, extension, content_type, "
    "size_bytes, status, created_at, updated_at, processed_path, error_message"
)


def utc_now() -> str:
    """Current time as text, e.g. "2026-09-24T01:30:00.123456+00:00"."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database (creating file and table if needed) in a transaction.

    The transaction is committed when the "with" block ends normally and
    rolled back if an error happens inside it.
    """
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db_path)) as connection:
            with connection:
                connection.execute(SCHEMA)
                yield connection
    except sqlite3.Error as error:
        logger.error("SQLite error: %s", type(error).__name__)
        raise StorageError("Could not access the document database.")
    except OSError as error:
        logger.error("Could not open the database folder: %s", type(error).__name__)
        raise StorageError("Could not access the document database.")


def initialize(db_path: Path) -> None:
    """Create the database file and the documents table if they do not exist."""
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
        return cursor.rowcount == 1


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
