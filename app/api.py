"""Minimal FastAPI web API for DocuBot AI.

Endpoints:
    GET  /health                          -> check that the API is running
    POST /upload                          -> validate one document and save it
    POST /documents/{document_id}/process -> extract and normalize its text
    GET  /documents/{document_id}         -> processing status and metadata
    POST /documents/{document_id}/chunk   -> split processed text into chunks
    GET  /documents/{document_id}/chunks  -> list the stored chunks
    POST /documents/{document_id}/embed   -> embed the chunks with the local model
    GET  /documents/{document_id}/embeddings -> embedding status (no vectors)
    POST /search                          -> exact vector search over valid chunk vectors

Processing, chunking, embedding and search are synchronous. Search returns
ranked chunks only: there is NO reranking and no RAG answer generation yet.

Run locally from the project root:

    .venv\\Scripts\\python.exe -m uvicorn app.api:app --reload
"""

import uuid
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.document_processing import database
from app.document_processing.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingConfig,
)
from app.document_processing.models import (
    AlreadyProcessingError,
    Chunk,
    ChunkingConflictError,
    ChunkSet,
    DocumentContentError,
    DocumentNotFoundError,
    DocumentNotProcessedError,
    DocumentRecord,
    InvalidChunkingConfigError,
    InvalidDocumentIdError,
    ProcessedOutputMissingError,
    ProcessingError,
    StorageError,
    StoredFileMissingError,
    UnsupportedFileTypeError,
)
from app.document_processing.processor import (
    chunk_document,
    find_document,
    process_document,
)
from app.embeddings.config import EmbeddingConfig
from app.embeddings.models import (
    EmbeddingConflictError,
    EmbeddingContract,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    EmbeddingModelUnavailableError,
    InvalidEmbeddingConfigError,
    NoChunksError,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.provider import EmbeddingProvider, FastEmbedProvider
from app.embeddings.service import embed_document, get_embedding_status
from app.search.models import (
    DEFAULT_TOP_K,
    MAX_QUERY_LENGTH,
    MAX_TOP_K,
    InvalidQueryVectorError,
    InvalidSearchQueryError,
    InvalidTopKError,
    SearchNotSupportedError,
    SearchResult,
    validate_query,
)
from app.search.service import search

# --- Configuration ----------------------------------------------------------

# Project root is the folder above app/. Everything is stored under <project>/storage.
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
PROCESSED_DIR = STORAGE_DIR / "processed"
DATABASE_PATH = STORAGE_DIR / "metadata" / "documents.db"
MODEL_CACHE_DIR = STORAGE_DIR / "model_cache"  # downloaded model files (git-ignored)

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


class ChunkSummaryResponse(BaseModel):
    """Chunking state of a document (no chunk text).

    The configuration fields are null when the document has no chunks.
    """

    document_id: str
    chunking_status: str  # "chunked" or "not_chunked"
    chunk_count: int
    chunking_version: int | None
    chunk_size: int | None
    chunk_overlap: int | None
    chunked_at: str | None

    @classmethod
    def from_chunk_set(cls, chunk_set: ChunkSet, **extra) -> "ChunkSummaryResponse":
        first = chunk_set.chunks[0] if chunk_set.chunks else None
        return cls(
            document_id=chunk_set.document_id,
            chunking_status=chunk_set.status.value,
            chunk_count=len(chunk_set.chunks),
            chunking_version=first.chunking_version if first else None,
            chunk_size=first.chunk_size if first else None,
            chunk_overlap=first.chunk_overlap if first else None,
            chunked_at=chunk_set.created_at,
            **extra,
        )


class ChunkResponse(BaseModel):
    """One chunk's metadata. text is null unless include_text=true was asked."""

    chunk_id: str
    chunk_index: int
    char_count: int
    source_locations: list[dict[str, int]]
    text: str | None

    @classmethod
    def from_chunk(cls, chunk: Chunk, include_text: bool) -> "ChunkResponse":
        return cls(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            char_count=chunk.char_count,
            source_locations=[dict(location) for location in chunk.source_locations],
            text=chunk.text if include_text else None,
        )


class ChunkListResponse(ChunkSummaryResponse):
    """All chunks of one document, in order."""

    source_filename: str
    file_type: str
    chunks: list[ChunkResponse]


class EmbeddingContractResponse(BaseModel):
    """Which contract produced (or would produce) the vectors. Never the vectors.

    The prefixes are shown so a reader knows what the model really reads:
    "passage: " + chunk text for documents, "query: " + question for searches.
    """

    document_id: str
    status: str  # "complete", "incomplete" or "no_chunks"
    model_name: str
    model_revision: str  # exact Hugging Face commit of the model files
    embedding_version: int
    dimension: int
    max_tokens: int  # the model reads at most this many tokens per chunk
    passage_prefix: str
    query_prefix: str
    dtype: str
    normalized: bool
    total_chunks: int

    @staticmethod
    def contract_fields(contract: EmbeddingContract) -> dict:
        # Every contract field, so the response can never silently lag behind
        # the contract (a test checks that the field sets match).
        return asdict(contract)


class EmbedResponse(EmbeddingContractResponse):
    """Returned after an embedding run."""

    embedded_count: int  # embedded in this run (missing + stale)
    skipped_count: int  # already had a valid embedding
    stale_reembedded_count: int  # part of embedded_count that replaced stale vectors


class EmbeddingStatusResponse(EmbeddingContractResponse):
    """Embedding state of a document for the current model contract."""

    embedded_chunks: int
    missing_embeddings: int
    stale_embeddings: int
    truncated_chunks: int  # the model only read the first part of these chunks


class SearchRequest(BaseModel):
    """A search question. Unknown fields and wrong types (e.g. "5" for top_k) are rejected."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"query": "What is FastAPI?", "top_k": 5}]},
    )

    query: str = Field(
        strict=True,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="The question. Surrounding whitespace is removed; it must not be blank.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        strict=True,
        ge=1,
        le=MAX_TOP_K,
        description="How many chunks to return at most.",
    )
    document_id: str | None = Field(
        default=None,
        strict=True,
        description="Search only this document (32 lower-case hex characters). "
        "Omit it to search every document.",
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        try:
            return validate_query(value)
        except InvalidSearchQueryError as error:
            raise ValueError(error.safe_message)


class SearchResultResponse(BaseModel):
    """One ranked chunk. The vector itself is never returned."""

    rank: int  # 1 = most similar
    score: float  # cosine similarity from -1 to 1 (not a probability)
    chunk_id: str
    document_id: str
    chunk_index: int
    source_filename: str
    source_locations: list[dict[str, int]]
    truncated: bool  # the model only read the first part of this chunk
    text: str

    @classmethod
    def from_result(cls, result: SearchResult) -> "SearchResultResponse":
        chunk = result.chunk
        return cls(
            rank=result.rank,
            score=result.score,
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            chunk_index=chunk.chunk_index,
            source_filename=chunk.source_filename,
            source_locations=[dict(location) for location in chunk.source_locations],
            truncated=result.truncated,
            text=chunk.text,
        )


class SearchResponse(BaseModel):
    query: str  # the trimmed question that was searched
    top_k: int
    document_id: str | None
    results: list[SearchResultResponse]  # best first; empty if nothing is searchable


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
    DocumentNotProcessedError: status.HTTP_409_CONFLICT,
    ProcessedOutputMissingError: status.HTTP_404_NOT_FOUND,
    InvalidChunkingConfigError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    ChunkingConflictError: status.HTTP_409_CONFLICT,
    InvalidEmbeddingConfigError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    NoChunksError: status.HTTP_409_CONFLICT,
    EmbeddingConflictError: status.HTTP_409_CONFLICT,
    EmbeddingModelUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    EmbeddingModelMismatchError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    UnsupportedEmbeddingModelError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    EmbeddingError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    EmbeddingDimensionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    InvalidSearchQueryError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    InvalidTopKError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    SearchNotSupportedError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    InvalidQueryVectorError: status.HTTP_500_INTERNAL_SERVER_ERROR,
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


@app.post("/documents/{document_id}/chunk")
def chunk(
    document_id: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> ChunkSummaryResponse:
    """Split a processed document into chunks and store them.

    Chunking again replaces the previous chunks of this document, so chunks
    are never duplicated. Only a summary is returned, never the chunk text.
    """
    # Invalid settings are rejected before any storage is touched.
    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunk_set = chunk_document(
        document_id,
        upload_dir=UPLOAD_DIR,
        processed_dir=PROCESSED_DIR,
        db_path=DATABASE_PATH,
        config=config,
    )
    return ChunkSummaryResponse.from_chunk_set(chunk_set)


@app.get("/documents/{document_id}/chunks")
def list_chunks(document_id: str, include_text: bool = False) -> ChunkListResponse:
    """Return the chunks of a document: IDs, order, sizes and source locations.

    Chunk text is only included when include_text=true is requested.
    """
    record = find_document(document_id, UPLOAD_DIR, DATABASE_PATH)
    chunk_set = database.get_chunk_set(DATABASE_PATH, record.document_id)
    return ChunkListResponse.from_chunk_set(
        chunk_set,
        source_filename=record.original_filename,
        file_type=record.extension,
        chunks=[ChunkResponse.from_chunk(c, include_text) for c in chunk_set.chunks],
    )


# --- Embeddings ---------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    """The one embedding provider of this process, created on first use.

    lru_cache keeps the same instance, so the model is loaded once per process
    and reused by every request. Tests replace it with a small fake model via
    app.dependency_overrides.
    """
    return FastEmbedProvider(EmbeddingConfig(cache_dir=MODEL_CACHE_DIR))


@app.post("/documents/{document_id}/embed")
def embed(
    document_id: str,
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    batch_size: int | None = None,
) -> EmbedResponse:
    """Embed a chunked document's chunks with the local model.

    Only chunks without a valid embedding are embedded, so running it again is
    cheap and never creates duplicates. Vectors are never returned.
    """
    summary = embed_document(
        document_id,
        upload_dir=UPLOAD_DIR,
        db_path=DATABASE_PATH,
        provider=provider,
        batch_size=batch_size,
    )
    return EmbedResponse(
        document_id=summary.document_id,
        status="complete",
        total_chunks=summary.total_chunks,
        embedded_count=summary.embedded_count,
        skipped_count=summary.skipped_count,
        stale_reembedded_count=summary.stale_reembedded_count,
        **EmbeddingContractResponse.contract_fields(summary.contract),
    )


@app.get("/documents/{document_id}/embeddings")
def embedding_status(
    document_id: str,
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> EmbeddingStatusResponse:
    """How many chunks have valid, missing or stale embeddings (no vectors)."""
    result = get_embedding_status(
        document_id, upload_dir=UPLOAD_DIR, db_path=DATABASE_PATH, contract=provider.contract
    )
    return EmbeddingStatusResponse(
        document_id=result.document_id,
        status=result.state.value,
        total_chunks=result.total_chunks,
        embedded_chunks=result.embedded_chunks,
        missing_embeddings=result.missing_embeddings,
        stale_embeddings=result.stale_embeddings,
        truncated_chunks=result.truncated_chunks,
        **EmbeddingContractResponse.contract_fields(result.contract),
    )


# --- Search -------------------------------------------------------------------


@app.post("/search")
def vector_search(
    request: SearchRequest,
    provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
) -> SearchResponse:
    """Find the chunks whose meaning is closest to the question (exact vector search).

    The question is embedded as "query: <question>" with the same local model
    that embedded the chunks, then compared with every chunk vector that is
    valid for the current embedding contract (stale vectors are ignored).
    Results are ranked by score (highest first), ties by chunk_id. Nothing
    is stored: not the question, not the query vector. Vectors are never returned.
    """
    result = search(
        request.query,
        db_path=DATABASE_PATH,
        provider=provider,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    return SearchResponse(
        query=result.query,
        top_k=result.top_k,
        document_id=result.document_id,
        results=[SearchResultResponse.from_result(item) for item in result.results],
    )
