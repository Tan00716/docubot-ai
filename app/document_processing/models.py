"""Data shapes shared by the document-processing pipeline.

This module only describes data. It does not read files, touch SQLite or
know anything about HTTP.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

# Bump this when the JSON layout of a processed document changes, so later
# stages (for example Chunking) can tell which layout they are reading.
PROCESSED_SCHEMA_VERSION = 1


class DocumentStatus(StrEnum):
    """Every allowed processing status. Never use free-form status strings."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkingStatus(StrEnum):
    """Whether a document currently has chunks stored.

    This is separate from DocumentStatus (processing). It is not stored as a
    column: a document is "chunked" exactly when it has rows in the chunks table.
    """

    NOT_CHUNKED = "not_chunked"
    CHUNKED = "chunked"


def parse_source_location(data: Any) -> dict[str, int]:
    """Check that a source location looks like {"page": 3}.

    Keys must be text and values whole numbers. Anything else means the
    stored data is corrupted, so ValueError is raised.
    """
    if not isinstance(data, dict) or not data:
        raise ValueError("Invalid source location.")
    for key, value in data.items():
        # bool is a subclass of int in Python, so True would pass as 1.
        if not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Invalid source location.")
    return dict(data)


@dataclass(frozen=True)
class Section:
    """One piece of extracted text plus where it came from in the source file.

    source_location examples:
        PDF:  {"page": 3}
        DOCX: {"paragraph": 12}
        TXT:  {"block": 2, "line_start": 5, "line_end": 9}
    """

    source_location: dict[str, int]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"source_location": dict(self.source_location), "text": self.text}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Section":
        if not isinstance(data, dict) or not isinstance(data.get("text"), str):
            raise ValueError("Invalid section.")
        return cls(
            source_location=parse_source_location(data.get("source_location")),
            text=data["text"],
        )


@dataclass(frozen=True)
class ProcessedDocument:
    """The common representation that every parser (PDF, DOCX, TXT) produces."""

    document_id: str
    source_filename: str
    file_type: str
    sections: tuple[Section, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROCESSED_SCHEMA_VERSION,
            "document_id": self.document_id,
            "source_filename": self.source_filename,
            "file_type": self.file_type,
            "sections": [section.to_dict() for section in self.sections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessedDocument":
        """Rebuild a document from its JSON form, rejecting anything malformed."""
        if not isinstance(data, dict):
            raise ValueError("Processed document must be a JSON object.")
        if data.get("schema_version") != PROCESSED_SCHEMA_VERSION:
            raise ValueError("Unsupported processed document schema version.")
        for field in ("document_id", "source_filename", "file_type"):
            if not isinstance(data.get(field), str):
                raise ValueError(f"Invalid processed document field: {field}.")
        if not isinstance(data.get("sections"), list):
            raise ValueError("Processed document sections must be a list.")
        return cls(
            document_id=data["document_id"],
            source_filename=data["source_filename"],
            file_type=data["file_type"],
            sections=tuple(Section.from_dict(item) for item in data["sections"]),
        )


@dataclass(frozen=True)
class Chunk:
    """One chunk: a small piece of a processed document, ready for embedding.

    text is exactly the string that a later stage will embed.

    source_locations lists every section that contributed text, in order. Each
    entry is that section's Batch 3 source_location plus the character range
    used from the section's text (end is exclusive, like Python slicing):
        {"page": 2, "char_start": 0, "char_end": 812}

    chunk_id is deterministic (see chunking.make_chunk_id), never random.
    """

    chunk_id: str
    document_id: str
    chunk_index: int  # 0 = first chunk of the document
    text: str
    char_count: int
    source_filename: str
    file_type: str
    source_locations: tuple[dict[str, int], ...]
    chunking_version: int
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class ChunkSet:
    """All chunks currently stored for one document (possibly none)."""

    document_id: str
    chunks: tuple[Chunk, ...]
    created_at: str | None  # when these chunks were stored; None if no chunks

    @property
    def status(self) -> ChunkingStatus:
        return ChunkingStatus.CHUNKED if self.chunks else ChunkingStatus.NOT_CHUNKED


@dataclass(frozen=True)
class DocumentRecord:
    """One row of the SQLite "documents" table."""

    document_id: str
    original_filename: str
    stored_filename: str
    extension: str
    content_type: str | None
    size_bytes: int
    status: DocumentStatus
    created_at: str
    updated_at: str
    processed_path: str | None  # relative to the processed-output folder
    error_message: str | None


# --- Errors -------------------------------------------------------------------
#
# Every error carries a "safe_message": a short sentence that is safe to show
# to API users and to store in the database. It never contains file paths,
# stack traces or document contents.


class ProcessingError(Exception):
    """Base class for every expected document-processing problem."""

    safe_message = "Document processing failed."

    def __init__(self, safe_message: str | None = None):
        if safe_message is not None:
            self.safe_message = safe_message
        super().__init__(self.safe_message)


class InvalidDocumentIdError(ProcessingError):
    safe_message = "Invalid document ID."


class DocumentNotFoundError(ProcessingError):
    safe_message = "Document not found."


class StoredFileMissingError(ProcessingError):
    safe_message = "The stored file for this document is missing."


class AlreadyProcessingError(ProcessingError):
    safe_message = "This document is already being processed."


class UnsupportedFileTypeError(ProcessingError):
    safe_message = "Unsupported file type. Supported: .docx, .pdf, .txt."


class DocumentContentError(ProcessingError):
    """The file itself could not be turned into text (corrupt, empty, ...)."""

    safe_message = "The document could not be read."


class StorageError(ProcessingError):
    """SQLite or the processed-output folder could not be read or written."""

    safe_message = "Could not access document storage."


class DocumentNotProcessedError(ProcessingError):
    """A later stage (e.g. Chunking) needs a document that is "completed"."""

    safe_message = "This document has not been processed yet."


class ProcessedOutputMissingError(ProcessingError):
    safe_message = (
        "The processed output for this document is missing. Process the document again."
    )


class InvalidChunkingConfigError(ProcessingError):
    safe_message = "Invalid chunking configuration."


class ChunkingConflictError(ProcessingError):
    """The document was re-processed while it was being chunked."""

    safe_message = "The document changed while it was being chunked. Please try again."
