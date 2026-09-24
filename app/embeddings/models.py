"""Data shapes for embeddings. This module only describes data.

Errors reuse the document-processing ProcessingError base class, so the API
turns them into safe {"detail": ...} responses the same way.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.document_processing.models import ProcessingError


@dataclass(frozen=True)
class EmbeddingContract:
    """Everything that must match for a stored vector to be usable.

    A vector is only meaningful together with the model that produced it.
    Two vectors may only be compared if their contracts are equal.
    """

    model_name: str
    embedding_version: int
    dimension: int  # how many numbers each vector has (384 for the default model)
    dtype: str  # "float32"
    normalized: bool  # True: every vector has length 1


@dataclass(frozen=True)
class EmbeddedText:
    """What the provider returns for one input text."""

    vector: tuple[float, ...]
    truncated: bool  # the model only read the first part of the text


@dataclass(frozen=True)
class NewEmbedding:
    """One vector ready to be stored for a chunk."""

    chunk_id: str
    text_sha256: str  # SHA-256 of exactly the text given to the model
    vector: tuple[float, ...]
    truncated: bool


@dataclass(frozen=True)
class EmbeddingRecord:
    """Metadata of one stored embedding (the vector itself is not loaded)."""

    chunk_id: str
    model_name: str
    embedding_version: int
    dimension: int
    dtype: str
    normalized: bool
    text_sha256: str
    truncated: bool
    created_at: str

    def is_valid_for(self, contract: EmbeddingContract, text_sha256: str) -> bool:
        """True only if this vector was made by `contract` from exactly this text."""
        return (
            self.model_name == contract.model_name
            and self.embedding_version == contract.embedding_version
            and self.dimension == contract.dimension
            and self.dtype == contract.dtype
            and self.normalized == contract.normalized
            and self.text_sha256 == text_sha256
        )


@dataclass(frozen=True)
class StoredEmbedding:
    """A stored embedding including its vector."""

    record: EmbeddingRecord
    vector: tuple[float, ...]


class EmbeddingState(StrEnum):
    NO_CHUNKS = "no_chunks"  # nothing to embed (document not chunked)
    INCOMPLETE = "incomplete"  # some chunks have no valid embedding
    COMPLETE = "complete"  # every chunk has a valid embedding


@dataclass(frozen=True)
class EmbeddingStatus:
    """How many chunks of a document have valid, missing or stale embeddings."""

    document_id: str
    contract: EmbeddingContract
    total_chunks: int
    embedded_chunks: int  # valid for the current contract and current text
    missing_embeddings: int  # no embedding stored at all
    stale_embeddings: int  # stored, but for another model/version/text
    truncated_chunks: int  # valid embeddings whose text was cut by the model

    @property
    def state(self) -> EmbeddingState:
        if self.total_chunks == 0:
            return EmbeddingState.NO_CHUNKS
        if self.embedded_chunks == self.total_chunks:
            return EmbeddingState.COMPLETE
        return EmbeddingState.INCOMPLETE


@dataclass(frozen=True)
class EmbeddingRunSummary:
    """What one embedding run did.

    total_chunks = embedded_count + skipped_count, and
    stale_reembedded_count is the part of embedded_count that replaced stale vectors.
    """

    document_id: str
    contract: EmbeddingContract
    total_chunks: int
    embedded_count: int
    skipped_count: int
    stale_reembedded_count: int


# --- Errors -------------------------------------------------------------------


class InvalidEmbeddingConfigError(ProcessingError):
    safe_message = "Invalid embedding configuration."


class UnsupportedEmbeddingModelError(ProcessingError):
    safe_message = "The configured embedding model is not supported."


class EmbeddingModelUnavailableError(ProcessingError):
    safe_message = (
        "The embedding model could not be loaded. On the first run it is downloaded "
        "once; check the internet connection and try again."
    )


class NoChunksError(ProcessingError):
    safe_message = "This document has no chunks yet. Chunk it first."


class EmbeddingError(ProcessingError):
    """The model failed or returned something unusable."""

    safe_message = "The embedding model failed to embed the text."


class EmbeddingDimensionError(EmbeddingError):
    safe_message = "The embedding model returned a vector with an unexpected dimension."


class EmbeddingConflictError(ProcessingError):
    """The chunks changed (re-chunked or re-processed) while embedding."""

    safe_message = "The document's chunks changed while it was being embedded. Please try again."
