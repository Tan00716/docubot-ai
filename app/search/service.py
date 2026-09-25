"""Exact vector search: the orchestration of one search request.

    validate query + top_k (+ document_id)
    -> load candidates whose vectors are VALID for the current contract
    -> nothing to search? return no results (the model is not even loaded)
    -> embed the question once ("query: " + question, same model and contract)
    -> check every vector, score all candidates at once (cosine = dot product)
    -> rank: score DESC, then chunk_id ASC -> keep the best top_k

"Exact" (brute force) means every candidate is compared with the query:
about N x D multiplications for N chunks of D numbers. This is correct and
simple, fine for local development and small collections, but it is not
meant for millions of chunks. An index (FAISS, a vector database) could
later replace load_candidates + cosine_similarities without changing the API.

Search is read-only: it never writes chunks, embeddings, documents or the
question itself. The question is not logged either.
"""

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from app.document_processing import database
from app.document_processing.models import DocumentNotFoundError, ProcessingError, StorageError
from app.document_processing.processor import validate_document_id
from app.embeddings.provider import EmbeddingProvider
from app.embeddings.storage import CORRUPTED_EMBEDDINGS_MESSAGE
from app.search import storage
from app.search.models import (
    DEFAULT_TOP_K,
    SCORE_DECIMALS,
    InvalidQueryVectorError,
    SearchCandidate,
    SearchNotSupportedError,
    SearchResult,
    SearchResults,
    validate_query,
    validate_top_k,
)
from app.search.similarity import (
    VectorValidationError,
    as_candidate_matrix,
    as_query_vector,
    cosine_similarities,
)

logger = logging.getLogger(__name__)

UNEXPECTED_SEARCH_ERROR_MESSAGE = "Unexpected error while searching."


def rank_top_k(scores: Sequence[float], chunk_ids: Sequence[str], top_k: int) -> list[int]:
    """Positions of the best top_k candidates, best first (pure function).

    Order: higher score first; equal scores -> smaller chunk_id first. The
    order therefore never depends on how the database returned the rows.
    Scores are compared after rounding to SCORE_DECIMALS, so two chunks whose
    scores differ only by float32 noise count as a tie and are ordered by ID.
    """
    if len(scores) != len(chunk_ids):
        raise ValueError("Every score needs exactly one chunk_id.")
    rounded = [round(float(score), SCORE_DECIMALS) for score in scores]
    order = sorted(range(len(rounded)), key=lambda i: (-rounded[i], chunk_ids[i]))
    return order[:top_k]


def _check_document_filter(document_id: str | None, db_path: Path) -> str | None:
    """Validate the optional filter (400 if malformed, 404 if unknown). Read-only."""
    if document_id is None:
        return None
    validate_document_id(document_id)
    if database.get_document(db_path, document_id) is None:
        raise DocumentNotFoundError()
    return document_id


def _score(
    provider: EmbeddingProvider, query: str, candidates: list[SearchCandidate]
) -> list[float]:
    contract = provider.contract
    try:
        matrix = as_candidate_matrix([c.vector for c in candidates], contract.dimension)
    except VectorValidationError as error:
        logger.error("Stored vector breaks the search contract: %s", error)
        raise StorageError(CORRUPTED_EMBEDDINGS_MESSAGE)

    # The provider adds the "query: " prefix itself (contract.query_input).
    raw_query_vector = provider.embed_query(query)
    try:
        query_vector = as_query_vector(raw_query_vector, contract.dimension)
    except VectorValidationError as error:
        logger.error("Query vector breaks the search contract: %s", error)
        raise InvalidQueryVectorError()
    return cosine_similarities(matrix, query_vector).tolist()


def search(
    query: str,
    *,
    db_path: Path,
    provider: EmbeddingProvider,
    top_k: int = DEFAULT_TOP_K,
    document_id: str | None = None,
) -> SearchResults:
    """Find the top_k chunks whose meaning is closest to `query`.

    Raises a ProcessingError subclass (with a safe_message) on any failure.
    """
    query = validate_query(query)
    top_k = validate_top_k(top_k)
    contract = provider.contract
    if not contract.normalized:
        # Without unit vectors, a dot product is not cosine similarity.
        raise SearchNotSupportedError()
    document_id = _check_document_filter(document_id, db_path)

    started = time.perf_counter()
    try:
        candidates, stale_count = storage.load_candidates(db_path, contract, document_id)
        if not candidates:
            scores: list[float] = []
        else:
            scores = _score(provider, query, candidates)
        chunk_ids = [candidate.chunk.chunk_id for candidate in candidates]
        best = rank_top_k(scores, chunk_ids, top_k)
    except ProcessingError:
        raise
    except Exception:
        # Full details go to the server log only, never to the API.
        logger.exception("Unexpected error while searching")
        raise ProcessingError(UNEXPECTED_SEARCH_ERROR_MESSAGE)

    results = tuple(
        SearchResult(
            rank=rank,
            score=round(scores[index], SCORE_DECIMALS),
            chunk=candidates[index].chunk,
            truncated=candidates[index].truncated,
        )
        for rank, index in enumerate(best, start=1)
    )
    # Counts and timing only: the question and the chunk texts are not logged.
    logger.info(
        "Search: %d candidates, %d stale skipped, %d results in %.1f ms",
        len(candidates), stale_count, len(results), (time.perf_counter() - started) * 1000,
    )
    return SearchResults(query=query, top_k=top_k, document_id=document_id, results=results)
