"""The embedding pipeline for one document.

    find record -> must be "completed" -> load chunks (chunk_index order)
    -> hash each chunk's text -> compare with stored embeddings
    -> embed only missing/stale chunks, batch by batch
    -> save each batch in its own short transaction

Repeating it is safe (idempotent): chunks that already have a valid
embedding for the current contract and current text are skipped. If a run
stops halfway, the finished batches stay saved and the next run continues.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.document_processing import database
from app.document_processing.models import (
    Chunk,
    DocumentNotProcessedError,
    DocumentStatus,
    ProcessingError,
)
from app.document_processing.processor import find_document
from app.embeddings import storage
from app.embeddings.config import validate_batch_size
from app.embeddings.models import (
    EmbeddingContract,
    EmbeddingError,
    EmbeddingRecord,
    EmbeddingRunSummary,
    EmbeddingStatus,
    NewEmbedding,
    NoChunksError,
)
from app.embeddings.provider import EmbeddingProvider
from app.embeddings.vectors import text_sha256

logger = logging.getLogger(__name__)

UNEXPECTED_EMBEDDING_ERROR_MESSAGE = "Unexpected error while embedding the document."


@dataclass(frozen=True)
class EmbeddingPlan:
    """Which chunks already have a usable vector and which need a new one."""

    valid: tuple[Chunk, ...]  # stored vector matches contract and text: skip
    missing: tuple[Chunk, ...]  # no vector stored yet
    stale: tuple[Chunk, ...]  # vector stored, but for another model/version/text

    @property
    def to_embed(self) -> list[Chunk]:
        """Missing and stale chunks, in chunk_index order."""
        return sorted([*self.missing, *self.stale], key=lambda chunk: chunk.chunk_index)


def plan_embeddings(
    chunks: Sequence[Chunk],
    records: dict[str, EmbeddingRecord],
    contract: EmbeddingContract,
) -> EmbeddingPlan:
    """Sort chunks into valid / missing / stale (pure function, no I/O)."""
    valid, missing, stale = [], [], []
    for chunk in chunks:
        record = records.get(chunk.chunk_id)
        if record is None:
            missing.append(chunk)
        elif record.is_valid_for(contract, text_sha256(chunk.text)):
            valid.append(chunk)
        else:
            stale.append(chunk)
    return EmbeddingPlan(valid=tuple(valid), missing=tuple(missing), stale=tuple(stale))


def _load_plan(
    document_id: str, db_path: Path, contract: EmbeddingContract
) -> tuple[tuple[Chunk, ...], EmbeddingPlan, dict[str, EmbeddingRecord]]:
    chunks = database.get_chunk_set(db_path, document_id).chunks
    records = storage.get_embedding_records(db_path, document_id)
    return chunks, plan_embeddings(chunks, records, contract), records


def get_embedding_status(
    document_id: str, *, upload_dir: Path, db_path: Path, contract: EmbeddingContract
) -> EmbeddingStatus:
    """Count valid, missing and stale embeddings (does not load the model)."""
    find_document(document_id, upload_dir, db_path)  # validates ID, 404 if unknown
    chunks, plan, records = _load_plan(document_id, db_path, contract)
    return EmbeddingStatus(
        document_id=document_id,
        contract=contract,
        total_chunks=len(chunks),
        embedded_chunks=len(plan.valid),
        missing_embeddings=len(plan.missing),
        stale_embeddings=len(plan.stale),
        truncated_chunks=sum(records[chunk.chunk_id].truncated for chunk in plan.valid),
    )


def _embed_batch(provider: EmbeddingProvider, batch: list[Chunk]) -> list[NewEmbedding]:
    # The text that is hashed is exactly the text that is given to the model.
    texts = [chunk.text for chunk in batch]
    results = provider.embed_documents(texts)
    if len(results) != len(batch):
        raise EmbeddingError()
    return [
        NewEmbedding(
            chunk_id=chunk.chunk_id,
            text_sha256=text_sha256(text),
            vector=result.vector,
            truncated=result.truncated,
        )
        for chunk, text, result in zip(batch, texts, results)
    ]


def embed_document(
    document_id: str,
    *,
    upload_dir: Path,
    db_path: Path,
    provider: EmbeddingProvider,
    batch_size: int | None = None,
) -> EmbeddingRunSummary:
    """Create or refresh the embeddings of one document's chunks.

    Raises a ProcessingError subclass (with a safe_message) on any failure.
    """
    batch_size = validate_batch_size(
        provider.config.batch_size if batch_size is None else batch_size
    )
    record = find_document(document_id, upload_dir, db_path)
    if record.status != DocumentStatus.COMPLETED:
        raise DocumentNotProcessedError()
    chunks, plan, _ = _load_plan(document_id, db_path, provider.contract)
    if not chunks:
        raise NoChunksError()

    to_embed = plan.to_embed
    try:
        for start in range(0, len(to_embed), batch_size):
            batch = to_embed[start:start + batch_size]
            storage.save_embeddings(db_path, provider.contract, _embed_batch(provider, batch))
    except ProcessingError as error:
        logger.warning("Embedding failed for document %s: %s", document_id, error.safe_message)
        raise
    except Exception:
        # Full details go to the server log only, never to the API.
        logger.exception("Unexpected error while embedding document %s", document_id)
        raise ProcessingError(UNEXPECTED_EMBEDDING_ERROR_MESSAGE)

    logger.info(
        "Embedded document %s: %d new, %d stale replaced, %d skipped",
        document_id, len(plan.missing), len(plan.stale), len(plan.valid),
    )
    return EmbeddingRunSummary(
        document_id=document_id,
        contract=provider.contract,
        total_chunks=len(chunks),
        embedded_count=len(to_embed),
        skipped_count=len(plan.valid),
        stale_reembedded_count=len(plan.stale),
    )
