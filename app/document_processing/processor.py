"""Run the processing pipeline for one uploaded document.

    find record -> uploaded/failed/completed -> processing
    -> read stored file -> parse + normalize -> save JSON -> completed
    (any error after "processing" -> failed, with a safe error message)

All paths are chosen by the server. The only outside value is document_id,
which must be 32 lower-case hex characters, so it can never contain "..",
"/" or "\\". Output is written only inside processed_dir and db_path.
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.document_processing import database
from app.document_processing.models import (
    AlreadyProcessingError,
    DocumentNotFoundError,
    DocumentRecord,
    DocumentStatus,
    InvalidDocumentIdError,
    ProcessedDocument,
    ProcessingError,
    StorageError,
    StoredFileMissingError,
    UnsupportedFileTypeError,
)
from app.document_processing.parsers import PARSERS, extract_sections

logger = logging.getLogger(__name__)

DOCUMENT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
UNEXPECTED_ERROR_MESSAGE = "Unexpected error while processing the document."


@dataclass(frozen=True)
class ProcessingResult:
    record: DocumentRecord
    document: ProcessedDocument


# --- Paths --------------------------------------------------------------------


def validate_document_id(document_id: str) -> str:
    if not DOCUMENT_ID_PATTERN.fullmatch(document_id):
        raise InvalidDocumentIdError()
    return document_id


def safe_child_path(folder: Path, filename: str) -> Path:
    """Return folder/filename, refusing anything that would leave the folder."""
    path = (folder / filename).resolve()
    if path.parent != folder.resolve():
        logger.error("Refused a storage path outside its folder.")
        raise StorageError("Invalid storage path.")
    return path


# --- Finding documents --------------------------------------------------------


def find_document(document_id: str, upload_dir: Path, db_path: Path) -> DocumentRecord:
    """Return the metadata record for an uploaded document.

    Files uploaded before metadata was recorded exist only in upload_dir.
    Those are registered on first use (their original filename is unknown,
    so the stored filename is used instead).
    """
    validate_document_id(document_id)

    record = database.get_document(db_path, document_id)
    if record is not None:
        return record

    for extension in PARSERS:
        stored_file = upload_dir / f"{document_id}{extension}"
        if stored_file.is_file():
            return database.create_document(
                db_path,
                document_id=document_id,
                original_filename=stored_file.name,
                stored_filename=stored_file.name,
                extension=extension,
                content_type=None,
                size_bytes=stored_file.stat().st_size,
            )

    raise DocumentNotFoundError()


# --- Processed output ---------------------------------------------------------


def save_processed_document(document: ProcessedDocument, processed_dir: Path) -> str:
    """Write the document as JSON and return its filename inside processed_dir.

    The same input always produces byte-for-byte the same file (no
    timestamps, fixed key order, "\\n" line endings). The file is written to
    a temporary name first and then renamed, so a half-written JSON file is
    never left under the final name.
    """
    filename = f"{validate_document_id(document.document_id)}.json"
    final_path = safe_child_path(processed_dir, filename)
    content = json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n"

    temp_path = None
    try:
        processed_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=processed_dir,
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(content)
        os.replace(temp_path, final_path)
    except OSError as error:
        logger.error("Could not write processed output: %s", type(error).__name__)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise StorageError("Could not save the processed document.")

    return filename


def load_processed_document(record: DocumentRecord, processed_dir: Path) -> ProcessedDocument:
    """Read back the processed JSON of a completed document (for later stages)."""
    if record.status != DocumentStatus.COMPLETED or not record.processed_path:
        raise ProcessingError("This document has not been processed yet.")

    path = safe_child_path(processed_dir, record.processed_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ProcessedDocument.from_dict(data)
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.error("Could not load processed output: %s", type(error).__name__)
        raise StorageError("Could not load the processed document.")


# --- Pipeline -----------------------------------------------------------------


def _read_stored_file(record: DocumentRecord, upload_dir: Path) -> bytes:
    path = safe_child_path(upload_dir, record.stored_filename)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise StoredFileMissingError()
    except OSError as error:
        logger.error("Could not read stored file: %s", type(error).__name__)
        raise StorageError("Could not read the stored file.")


def _build_processed_document(record: DocumentRecord, upload_dir: Path) -> ProcessedDocument:
    if record.extension not in PARSERS:
        raise UnsupportedFileTypeError()

    data = _read_stored_file(record, upload_dir)
    sections = extract_sections(data, record.extension)
    return ProcessedDocument(
        document_id=record.document_id,
        source_filename=record.original_filename,
        file_type=record.extension,
        sections=tuple(sections),
    )


def _record_failure(db_path: Path, document_id: str, message: str) -> None:
    try:
        database.mark_failed(db_path, document_id, message)
    except StorageError:
        # Do not hide the original problem. The row stays in "processing"
        # and becomes processable again after STALE_PROCESSING_AFTER.
        logger.error("Could not record failure for document %s", document_id)


def process_document(
    document_id: str,
    *,
    upload_dir: Path,
    processed_dir: Path,
    db_path: Path,
) -> ProcessingResult:
    """Process one document synchronously and return its updated record.

    Raises a ProcessingError subclass (with a safe_message) on any failure.
    """
    record = find_document(document_id, upload_dir, db_path)

    if not database.claim_for_processing(db_path, document_id):
        raise AlreadyProcessingError()
    logger.info("Processing document %s (%s)", document_id, record.extension)

    try:
        document = _build_processed_document(record, upload_dir)
        processed_path = save_processed_document(document, processed_dir)
        database.mark_completed(db_path, document_id, processed_path)
    except ProcessingError as error:
        logger.warning(
            "Processing failed for document %s: %s", document_id, error.safe_message
        )
        _record_failure(db_path, document_id, error.safe_message)
        raise
    except Exception:
        # Full details go to the server log only, never to the API or database.
        logger.exception("Unexpected error while processing document %s", document_id)
        _record_failure(db_path, document_id, UNEXPECTED_ERROR_MESSAGE)
        raise ProcessingError(UNEXPECTED_ERROR_MESSAGE)

    logger.info(
        "Processed document %s: %d sections", document_id, len(document.sections)
    )
    updated_record = database.get_document(db_path, document_id)
    if updated_record is None:
        raise StorageError("Could not read the document record.")
    return ProcessingResult(record=updated_record, document=document)
