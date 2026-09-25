"""Data shapes, limits and errors for vector search. This module only describes data.

Errors reuse the ProcessingError base class, so the API turns them into safe
{"detail": ...} responses the same way as every other stage.
"""

from dataclasses import dataclass

from app.document_processing.models import Chunk, ProcessingError

# How many results a search returns when the caller does not say.
DEFAULT_TOP_K = 5
# Upper limit for top_k: a caller cannot ask for (and make the server sort
# and serialize) an unbounded number of chunks.
MAX_TOP_K = 50
# Longest accepted question, in characters: a cheap first bound that keeps
# huge inputs away from the tokenizer. The REAL limit is the model's token
# limit (512 tokens, "query: " prefix included), checked with the model's own
# tokenizer when the question is embedded: a longer question is rejected
# (QueryTooLongError) instead of being silently cut. 4000 characters can be
# more than 512 tokens (Chinese text is roughly 1.5 characters per token).
MAX_QUERY_LENGTH = 4000
# Scores are rounded to this many decimals ONLY in the API response. Ranking
# always uses the full-precision score (see service.rank_top_k).
SCORE_DECIMALS = 6


@dataclass(frozen=True)
class SearchCandidate:
    """One chunk that may take part in a search: its vector is valid for the current contract."""

    chunk: Chunk
    truncated: bool  # the model only read the first part of this chunk
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SearchResult:
    """One ranked search hit (no vector: it never leaves the search layer)."""

    rank: int  # 1 = most similar
    score: float  # cosine similarity at full precision (rounded only by the API)
    chunk: Chunk
    truncated: bool


@dataclass(frozen=True)
class SearchResults:
    query: str  # the trimmed question that was embedded (never stored)
    top_k: int
    document_id: str | None
    results: tuple[SearchResult, ...]


# --- Errors -------------------------------------------------------------------


class InvalidSearchQueryError(ProcessingError):
    safe_message = f"query must be a non-empty string of at most {MAX_QUERY_LENGTH} characters."


class InvalidTopKError(ProcessingError):
    safe_message = f"top_k must be a whole number from 1 to {MAX_TOP_K}."


class SearchNotSupportedError(ProcessingError):
    """The configured embedding contract cannot be searched with a dot product."""

    safe_message = "Vector search requires normalized embeddings."


class InvalidQueryVectorError(ProcessingError):
    """The model returned a query vector that breaks the contract (a provider bug)."""

    safe_message = "The embedding model returned an invalid query vector."


# --- Validation ---------------------------------------------------------------


def validate_query(query: object) -> str:
    """Return the question without surrounding whitespace, or raise InvalidSearchQueryError."""
    if not isinstance(query, str) or len(query) > MAX_QUERY_LENGTH:
        raise InvalidSearchQueryError()
    trimmed = query.strip()
    if not trimmed:
        raise InvalidSearchQueryError()
    return trimmed


def validate_top_k(top_k: object) -> int:
    """Return top_k if it is a whole number from 1 to MAX_TOP_K."""
    # bool is a subclass of int in Python; True must not count as 1.
    is_whole_number = isinstance(top_k, int) and not isinstance(top_k, bool)
    if not is_whole_number or not 1 <= top_k <= MAX_TOP_K:
        raise InvalidTopKError()
    return top_k
