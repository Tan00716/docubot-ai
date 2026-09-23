"""Minimal FastAPI web API for DocuBot AI.

Endpoints:
    GET  /health  -> check that the API is running
    POST /upload  -> validate one document and save it to local storage

Uploaded files are only stored. They are NOT parsed, chunked or indexed yet.

Run locally from the project root:

    .venv\\Scripts\\python.exe -m uvicorn app.api:app --reload
"""

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Configuration ----------------------------------------------------------

# Project root is the folder above app/. Uploads go to <project>/storage/uploads.
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "storage" / "uploads"

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


app = FastAPI(title="DocuBot AI API")


# --- Error handling -----------------------------------------------------------


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Hide internal details (stack traces, paths) from API users."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error."},
    )


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

    return UploadResponse(
        file_id=file_id,
        filename=filename,
        stored_filename=stored_filename,
        extension=extension,
        content_type=file.content_type,
        size_bytes=len(content),
        status="stored",
    )
