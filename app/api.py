"""Minimal FastAPI web API for DocuBot AI.

Endpoints:
    GET  /health                          -> check that the API is running
    POST /upload                          -> validate one document and save it
    POST /documents/{document_id}/process -> extract and normalize its text
    GET  /documents/{document_id}         -> processing status and metadata

Processing is synchronous. Documents are NOT chunked or indexed yet.

Run locally from the project root:

    .venv\\Scripts\\python.exe -m uvicorn app.api:app --reload
"""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.document_processing import database
from app.document_processing.models import (
    AlreadyProcessingError,
    DocumentContentError,
    DocumentNotFoundError,
    DocumentRecord,
    InvalidDocumentIdError,
    ProcessingError,
    StorageError,
    StoredFileMissingError,
    UnsupportedFileTypeError,
)
from app.document_processing.processor import find_document, process_document

# --- Configuration ----------------------------------------------------------

# Project root is the folder above app/. Everything is stored under <project>/storage.
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
DATABASE_PATH = STORAGE_DIR / "metadata" / "documents.db"

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_FILENAME_LENGTH = 255
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# --- Response schema ----------------------------------------------------------


class UploadResponse(BaseModel):
    """Metadata returned after a successful upload (no filesystem paths)."""

    file_id: str
    filename: str
    stored_filename: str
    extension: str
    content_type: str | None
    size_bytes: int
    status: str


class DocumentResponse(BaseModel):
    """Document metadata and processing status (never the extracted text)."""

    document_id: str
    original_filename: str
    file_type: str
    content_type: str | None
    size_bytes: int
    status: str
    created_at: str
    updated_at: str
    processed_path: str | None  # filename inside storage/processed/
    error_message: str | None

    @classmethod
    def from_record(cls, record: DocumentRecord, **extra) -> "DocumentResponse":
        return cls(
            document_id=record.document_id,
            original_filename=record.original_filename,
            file_type=record.extension,
            content_type=record.content_type,
            size_bytes=record.size_bytes,
            status=record.status.value,
            created_at=record.created_at,
            updated_at=record.updated_at,
            processed_path=record.processed_path,
            error_message=record.error_message,
            **extra,
        )


class ProcessResponse(DocumentResponse):
    """Returned after successful processing."""

    section_count: int


app = FastAPI(title="DocuBot AI API")


# --- Error handling -----------------------------------------------------------


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Hide internal details (stack traces, paths) from API users."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


# Which HTTP status each processing error becomes. Only the error's
# safe_message is returned to the user.
PROCESSING_ERROR_STATUS = {
    InvalidDocumentIdError: status.HTTP_400_BAD_REQUEST,
    DocumentNotFoundError: status.HTTP_404_NOT_FOUND,
    StoredFileMissingError: status.HTTP_404_NOT_FOUND,
    AlreadyProcessingError: status.HTTP_409_CONFLICT,
    UnsupportedFileTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    DocumentContentError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    StorageError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


@app.exception_handler(ProcessingError)
async def handle_processing_error(request: Request, exc: ProcessingError) -> JSONResponse:
    status_code = PROCESSING_ERROR_STATUS.get(
        type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR
    )
    return JSONResponse(status_code=status_code, content={"detail": exc.safe_message})


def api_error(status_code: int, message: str) -> HTTPException:
    """Create a simple error response: {"detail": message}."""
    return HTTPException(status_code=status_code, detail=message)


# --- Validation helpers -------------------------------------------------------


def validate_filename(filename: str | None) -> str:
    """Check the user-provided filename. It is only kept as metadata."""
    if not filename:
        raise api_error(status.HTTP_400_BAD_REQUEST, "Filename is missing.")

    if len(filename) > MAX_FILENAME_LENGTH:
        raise api_error(status.HTTP_400_BAD_REQUEST, "Filename is too long.")

    # Reject anything that looks like a path ("../x.txt", "/etc/x.txt", "a\\b.txt")
    # or contains control characters.
    has_path_separator = "/" in filename or "\\" in filename
    has_control_char = any(ord(char) < 32 for char in filename)
    if has_path_separator or has_control_char or filename in {".", ".."}:
        raise api_error(status.HTTP_400_BAD_REQUEST, "Invalid filename.")

    return filename


def get_allowed_extension(filename: str) -> str:
    """Return the lower-case extension, or raise if it is not supported."""
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise api_error(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported file type. Allowed: {allowed}.",
        )
    return extension


def read_limited_content(file: UploadFile) -> bytes:
    """Read the file, but never more than MAX_UPLOAD_SIZE_BYTES + 1 bytes."""
    content = file.file.read(MAX_UPLOAD_SIZE_BYTES + 1)

    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise api_error(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"File is too large. Maximum size is {MAX_UPLOAD_SIZE_BYTES} bytes.",
        )
    if len(content) == 0:
        raise api_error(status.HTTP_400_BAD_REQUEST, "File is empty.")

    return content


# --- Endpoints ----------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    """Simple check that the API process is up."""
    return {"status": "ok"}


# This is a normal "def" (not "async def") on purpose: FastAPI runs it in a
# worker thread, so the blocking disk read/write below does not freeze the server.
@app.post("/upload", status_code=status.HTTP_201_CREATED)
def upload(file: Annotated[UploadFile, File()]) -> UploadResponse:
    """Validate one uploaded document and save it to local storage."""
    filename = validate_filename(file.filename)
    extension = get_allowed_extension(filename)
    content = read_limited_content(file)

    # The server chooses the stored name. The user's filename is NEVER used
    # as part of the filesystem path, so it cannot point outside UPLOAD_DIR.
    file_id = uuid.uuid4().hex
    stored_filename = f"{file_id}{extension}"

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # "xb" = create a new file in binary mode; fail instead of overwriting.
        with open(UPLOAD_DIR / stored_filename, "xb") as destination:
            destination.write(content)
    except OSError:
        raise api_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not store the file."
        )

    # Record the upload so it can be processed later (status "uploaded").
    try:
        database.create_document(
            DATABASE_PATH,
            document_id=file_id,
            original_filename=filename,
            stored_filename=stored_filename,
            extension=extension,
            content_type=file.content_type,
            size_bytes=len(content),
        )
    except StorageError:
        # Keep files and metadata consistent: no record means no stored file.
        (UPLOAD_DIR / stored_filename).unlink(missing_ok=True)
        raise api_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "Could not store the file."
        )

    return UploadResponse(
        file_id=file_id,
        filename=filename,
        stored_filename=stored_filename,
        extension=extension,
        content_type=file.content_type,
        size_bytes=len(content),
        status="stored",
    )


@app.post("/documents/{document_id}/process")
def process(document_id: str) -> ProcessResponse:
    """Extract, normalize and store the text of an uploaded document.

    Runs synchronously: the response is sent when processing has finished.
    Errors are turned into safe responses by handle_processing_error.
    """
    result = process_document(
        document_id,
        upload_dir=UPLOAD_DIR,
        processed_dir=PROCESSED_DIR,
        db_path=DATABASE_PATH,
    )
    return ProcessResponse.from_record(
        result.record, section_count=len(result.document.sections)
    )


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> DocumentResponse:
    """Return a document's metadata and processing status."""
    record = find_document(document_id, UPLOAD_DIR, DATABASE_PATH)
    return DocumentResponse.from_record(record)
