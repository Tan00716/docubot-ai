"""Embedding configuration: the ONE place that names the embedding model.

Everything that decides what a stored vector means lives here: the model,
its exact file revision, its input prefixes, EMBEDDING_VERSION and the
normalization setting. Changing any of them changes the "embedding
contract": every vector stored under the old contract becomes stale and is
regenerated the next time the document is embedded. Vectors from different
contracts are never mixed.
"""

from dataclasses import dataclass
from pathlib import Path

from app.embeddings.models import InvalidEmbeddingConfigError, UnsupportedEmbeddingModelError


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Facts about one embedding model, copied from its official Hugging Face files.

    The provider checks the downloaded files against these values, so a
    wrong entry here fails loudly instead of producing wrong vectors.
    """

    name: str  # the model's Hugging Face repository, e.g. "intfloat/multilingual-e5-small"
    revision: str  # exact git commit of that repository: always the same files
    model_file: str  # the ONNX model inside the repository
    dimension: int  # how many numbers each vector has
    max_tokens: int  # the tokenizer cuts every input after this many tokens
    pooling: str  # how the per-token vectors become one vector ("mean")
    passage_prefix: str  # put in front of every document chunk before embedding
    query_prefix: str  # put in front of every search question before embedding
    license: str


# The project's current embedding model: a development baseline, not a final
# production choice. Values checked on 2026-09-24 against the official repo:
#   config.json            hidden_size = 384
#   tokenizer_config.json  model_max_length = 512
#   1_Pooling/config.json  pooling_mode_mean_tokens = true (then Normalize)
#   model card             license MIT, 94 languages, inputs must start with
#                          "query: " or "passage: "
MULTILINGUAL_E5_SMALL = EmbeddingModelSpec(
    name="intfloat/multilingual-e5-small",
    revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
    model_file="onnx/model.onnx",  # full-precision float32 ONNX (about 470 MB)
    dimension=384,
    max_tokens=512,
    pooling="mean",
    passage_prefix="passage: ",
    query_prefix="query: ",
    license="mit",
)

# Only these models can be configured. A model name can therefore never point
# at arbitrary files: each one has a pinned revision and checked facts.
SUPPORTED_MODELS: dict[str, EmbeddingModelSpec] = {
    MULTILINGUAL_E5_SMALL.name: MULTILINGUAL_E5_SMALL,
}

DEFAULT_MODEL_NAME = MULTILINGUAL_E5_SMALL.name

# Bump this when the way DocuBot turns text into stored vectors changes.
# 1 = paraphrase-multilingual-MiniLM-L12-v2 (128 tokens, no prefix), Batch 5
# 2 = multilingual-e5-small with "passage: " / "query: " prefixes, Batch 5A
EMBEDDING_VERSION = 2

# How many chunks are sent to the model at once. Small batches keep memory
# low on a 16 GB CPU-only laptop; bigger batches are only a little faster.
DEFAULT_BATCH_SIZE = 16
MAX_BATCH_SIZE = 64

# Stored vectors are always 32-bit floats (4 bytes per number).
VECTOR_DTYPE = "float32"


def get_model_spec(model_name: str) -> EmbeddingModelSpec:
    """The spec of a supported model, or UnsupportedEmbeddingModelError."""
    spec = SUPPORTED_MODELS.get(model_name)
    if spec is None:
        raise UnsupportedEmbeddingModelError()
    return spec


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
    Whether model_name is supported is checked when the provider is created.
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
