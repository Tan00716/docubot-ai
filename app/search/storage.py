"""Read search candidates from SQLite (read-only).

One query loads every stored embedding together with its chunk (and, with a
document filter, only that document's rows). The existing row parsers and
the existing validity rule are reused, so search can never disagree with the
embedding status endpoint about which vectors are usable:

    database.to_chunk()               - chunk row -> Chunk (checks the stored JSON)
    storage.to_embedding_record()     - embedding row -> EmbeddingRecord (checks types)
    EmbeddingRecord.is_valid_for()    - THE rule: contract fields + both text hashes
    vectors.deserialize_vector()      - little-endian float32, exact size, finite

Stale vectors (another model, revision, version, prefix, dimension, token
limit, normalization, or text) are skipped without even reading their bytes.
A vector that passes the contract check but cannot be read is corrupted
data: the search stops with a safe StorageError instead of silently
returning results without it.

Nothing is written: no UPDATE, INSERT or DELETE is ever run here.
"""

import logging
from pathlib import Path

from app.document_processing.database import (
    CHUNK_SELECT_COLUMN_COUNT,
    CHUNK_SELECT_COLUMNS,
    connect,
    to_chunk,
)
from app.document_processing.models import StorageError
from app.embeddings.models import EmbeddingContract
from app.embeddings.storage import (
    CORRUPTED_EMBEDDINGS_MESSAGE,
    RECORD_COLUMNS,
    to_embedding_record,
)
from app.embeddings.vectors import deserialize_vector
from app.search.models import SearchCandidate

logger = logging.getLogger(__name__)

# Every value comes from a "?" placeholder; only these fixed column lists are
# pasted into the SQL. ORDER BY makes the candidate order independent of how
# SQLite happens to store the rows (the ranking does not rely on it either).
_CANDIDATES_SQL = (
    f"SELECT {CHUNK_SELECT_COLUMNS}, {RECORD_COLUMNS}, e.vector "
    "FROM embeddings AS e "
    "JOIN chunks AS c ON c.chunk_id = e.chunk_id "
    "JOIN documents AS d ON d.document_id = c.document_id"
)
_ALL_CANDIDATES_SQL = f"{_CANDIDATES_SQL} ORDER BY c.chunk_id"
_DOCUMENT_CANDIDATES_SQL = f"{_CANDIDATES_SQL} WHERE c.document_id = ? ORDER BY c.chunk_id"


def load_candidates(
    db_path: Path, contract: EmbeddingContract, document_id: str | None = None
) -> tuple[list[SearchCandidate], int]:
    """Chunks whose stored vector is valid for `contract`, and how many were stale.

    document_id (already validated by the caller) limits the search to one
    document; None searches every document.
    """
    with connect(db_path) as connection:
        if document_id is None:
            rows = connection.execute(_ALL_CANDIDATES_SQL).fetchall()
        else:
            rows = connection.execute(_DOCUMENT_CANDIDATES_SQL, (document_id,)).fetchall()

    candidates: list[SearchCandidate] = []
    stale_count = 0
    for row in rows:
        chunk_row = row[:CHUNK_SELECT_COLUMN_COUNT]
        record_row, blob = row[CHUNK_SELECT_COLUMN_COUNT:-1], row[-1]
        try:
            chunk = to_chunk(chunk_row)
            record = to_embedding_record(record_row)
            if not record.is_valid_for(contract, chunk.text):
                stale_count += 1
                continue
            vector = deserialize_vector(blob, contract.dimension)
        except (ValueError, TypeError) as error:
            # chunk_id and the error type go to the server log (no vector
            # values, no text); the API user only sees the safe message.
            logger.error("Corrupted search candidate %r: %s", row[0], type(error).__name__)
            raise StorageError(CORRUPTED_EMBEDDINGS_MESSAGE)
        candidates.append(SearchCandidate(chunk=chunk, truncated=record.truncated, vector=vector))
    return candidates, stale_count
