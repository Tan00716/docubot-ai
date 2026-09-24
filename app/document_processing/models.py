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
        return cls(source_location=dict(data["source_location"]), text=data["text"])


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
        if data.get("schema_version") != PROCESSED_SCHEMA_VERSION:
            raise ValueError("Unsupported processed document schema version.")
        return cls(
            document_id=data["document_id"],
            source_filename=data["source_filename"],
            file_type=data["file_type"],
            sections=tuple(Section.from_dict(item) for item in data["sections"]),
        )


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
