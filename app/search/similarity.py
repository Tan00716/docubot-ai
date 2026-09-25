"""Vector maths for search: cosine similarity of unit vectors (NumPy).

Cosine similarity measures the angle between two vectors: 1 = same
direction, 0 = unrelated (at right angles), -1 = opposite. In general

    cosine(a, b) = dot(a, b) / (|a| * |b|)

Every stored vector and every query vector has length 1 (the embedding
contract says normalized = true), so |a| * |b| = 1 and cosine(a, b) is just
dot(a, b). This module CHECKS that promise instead of assuming it: a vector
that is not of length 1 is rejected, never silently re-normalized, because
that would hide corrupted data or a broken provider.

This module knows nothing about SQLite, HTTP or the model.
"""

from collections.abc import Sequence

import numpy as np

# How far a vector's length may be from 1.0. Stored vectors are float32
# roundings of exact unit vectors, which moves their length by about 1e-7;
# anything much further away is not a unit vector.
UNIT_NORM_TOLERANCE = 1e-4


class VectorValidationError(ValueError):
    """A vector breaks the search contract (size, NaN/infinity, or length != 1).

    The message is for the server log only; callers turn it into a safe error.
    """


def _check_unit_rows(matrix: np.ndarray, what: str) -> None:
    if not np.all(np.isfinite(matrix)):
        row = int(np.argmin(np.all(np.isfinite(matrix), axis=1)))
        raise VectorValidationError(f"{what} {row} contains NaN or infinity.")
    lengths = np.linalg.norm(matrix, axis=1)
    bad_rows = np.flatnonzero(np.abs(lengths - 1.0) > UNIT_NORM_TOLERANCE)
    if bad_rows.size:
        row = int(bad_rows[0])
        raise VectorValidationError(f"{what} {row} is not normalized (length {lengths[row]:.6g}).")


def as_query_vector(values: Sequence[float], dimension: int) -> np.ndarray:
    """The query vector as a float64 array, after checking size, finiteness and length 1."""
    try:
        query = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise VectorValidationError("Query vector is not a list of numbers.")
    if query.shape != (dimension,):
        raise VectorValidationError(f"Query vector has shape {query.shape}, expected ({dimension},).")
    _check_unit_rows(query.reshape(1, dimension), "Query vector")
    return query


def as_candidate_matrix(vectors: Sequence[Sequence[float]], dimension: int) -> np.ndarray:
    """Stack candidate vectors into one (N, dimension) float64 matrix, checking every row.

    Building the matrix once lets all N similarities be computed in a single
    matrix-vector product instead of N separate Python loops.
    """
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except (TypeError, ValueError):  # rows of different lengths, non-numbers
        raise VectorValidationError("Candidate vectors do not form a matrix.")
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != dimension:
        raise VectorValidationError(
            f"Candidate matrix has shape {matrix.shape}, expected (N >= 1, {dimension})."
        )
    _check_unit_rows(matrix, "Candidate vector")
    return matrix


def cosine_similarities(candidates: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of the query with every candidate row (both already checked).

    Because every vector has length 1, cosine similarity = dot product, and
    `candidates @ query` computes all N dot products at once: O(N x D) work.
    """
    if candidates.ndim != 2 or query.ndim != 1 or candidates.shape[1] != query.shape[0]:
        raise VectorValidationError("Candidate matrix and query vector do not fit together.")
    scores = candidates @ query
    # Invariant: the dot product of two unit vectors lies in [-1, 1] (plus
    # rounding). Anything else means the checks above were bypassed.
    if not np.all(np.isfinite(scores)) or np.any(np.abs(scores) > 1.0 + UNIT_NORM_TOLERANCE):
        raise VectorValidationError("Similarity scores are outside [-1, 1].")
    return scores
