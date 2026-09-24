"""Embedding configuration: the ONE place that names the embedding model.

Changing DEFAULT_MODEL_NAME or EMBEDDING_VERSION changes the "embedding
contract": every stored vector made under the old contract becomes stale and
is regenerated the next time the document is embedded. Vectors from different
models are never mixed.
"""

from dataclasses import dataclass
from pathlib import Path

from app.embeddings.models import InvalidEmbeddingConfigError

# The project's initial embedding model (not a final production choice):
# multilingual (Chinese + English), 384 dimensions, small, runs on CPU.
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Bump this when the way DocuBot turns model output into stored vectors
# changes (for example normalization or dtype), even with the same model.
EMBEDDING_VERSION = 1

# How many chunks are sent to the model at once. Small batches keep memory
# low on a 16 GB CPU-only laptop; bigger batches are only a little faster.
DEFAULT_BATCH_SIZE = 16
MAX_BATCH_SIZE = 64

# Stored vectors are always 32-bit floats (4 bytes per number).
VECTOR_DTYPE = "float32"


def _is_whole_number(value: object) -> bool:
    # bool is a subclass of int in Python; True must not count as 1.
    return isinstance(value, int) and not isinstance(value, bool)


def validate_batch_size(batch_size: object) -> int:
    """Return batch_size if it is a whole number from 1 to MAX_BATCH_SIZE."""
    if not _is_whole_number(batch_size) or not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise InvalidEmbeddingConfigError(
            f"batch_size must be a whole number from 1 to {MAX_BATCH_SIZE}."
        )
    return batch_size


@dataclass(frozen=True)
class EmbeddingConfig:
    """Which model to use and how. Invalid values are rejected on creation.

    cache_dir is where the downloaded model files are kept (git-ignored).
    """

    cache_dir: Path
    model_name: str = DEFAULT_MODEL_NAME
    embedding_version: int = EMBEDDING_VERSION
    normalize_embeddings: bool = True
    batch_size: int = DEFAULT_BATCH_SIZE

    def __post_init__(self) -> None:
        if not isinstance(self.cache_dir, Path):
            raise InvalidEmbeddingConfigError("cache_dir must be a Path.")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise InvalidEmbeddingConfigError("model_name must be a non-empty string.")
        if not _is_whole_number(self.embedding_version) or self.embedding_version <= 0:
            raise InvalidEmbeddingConfigError(
                "embedding_version must be a positive whole number."
            )
        if not isinstance(self.normalize_embeddings, bool):
            raise InvalidEmbeddingConfigError("normalize_embeddings must be true or false.")
        validate_batch_size(self.batch_size)
