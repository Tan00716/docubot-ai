"""Pure helpers for vectors: hashing, normalization and float32 bytes.

Only the Python standard library is used, so these functions are easy to
test and behave the same everywhere.
"""

import hashlib
import math
import struct
from collections.abc import Sequence

# "<f" = little-endian 32-bit float. The byte order is fixed explicitly so a
# stored vector means the same thing on every computer.
_FLOAT32 = "<f"
FLOAT32_BYTES = struct.calcsize(_FLOAT32)  # 4


def text_sha256(text: str) -> str:
    """SHA-256 (hex) of a text, as UTF-8.

    Used for two fingerprints of every stored vector: the original chunk text
    (to notice when a chunk changed) and the exact model input with its prefix
    (to prove which input produced the vector). It is NOT an ID and has
    nothing to do with security here.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check_finite(values: Sequence[float]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Vector contains NaN or infinity.")


def l2_normalize(values: Sequence[float]) -> tuple[float, ...]:
    """Scale a vector to length 1 (its direction stays the same).

    After normalization, cosine similarity is just the dot product. This makes
    later similarity maths simpler; it says nothing about retrieval quality.
    """
    _check_finite(values)
    # math.fsum adds exactly, so the result does not depend on summation order.
    length = math.sqrt(math.fsum(value * value for value in values))
    if length == 0.0:
        raise ValueError("A zero vector cannot be normalized.")
    return tuple(value / length for value in values)


def to_float32(values: Sequence[float]) -> tuple[float, ...]:
    """Round every number to the nearest 32-bit float (what will be stored)."""
    count = len(values)
    return struct.unpack(f"<{count}f", struct.pack(f"<{count}f", *values))


def serialize_vector(values: Sequence[float], dimension: int) -> bytes:
    """Pack a vector into a compact float32 BLOB (4 bytes per number)."""
    if len(values) != dimension:
        raise ValueError(f"Vector has {len(values)} numbers, expected {dimension}.")
    _check_finite(values)
    return struct.pack(f"<{dimension}f", *values)


def deserialize_vector(blob: bytes, dimension: int) -> tuple[float, ...]:
    """Unpack a float32 BLOB, rejecting anything with the wrong size."""
    if not isinstance(blob, bytes) or len(blob) != dimension * FLOAT32_BYTES:
        raise ValueError("Stored vector has the wrong size.")
    values = struct.unpack(f"<{dimension}f", blob)
    _check_finite(values)
    return values
