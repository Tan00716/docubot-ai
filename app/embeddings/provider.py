"""The local embedding model (FastEmbed + ONNX Runtime on the CPU).

This is the ONLY module that imports fastembed. Everything outside it works
with plain Python tuples of floats, so the model can be replaced later
without touching storage or the API.

Lifecycle: creating a FastEmbedProvider is cheap. The model is loaded the
first time text is embedded, then reused for every later call (never once
per chunk). On the very first load FastEmbed downloads the model files into
config.cache_dir; after that, loading uses only the local files. Embedding
itself always runs locally: no hosted API is called.
"""

import logging
import threading
from collections.abc import Sequence
from typing import Any, Protocol

from fastembed import TextEmbedding

from app.embeddings.config import VECTOR_DTYPE, EmbeddingConfig
from app.embeddings.models import (
    EmbeddedText,
    EmbeddingContract,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingModelUnavailableError,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.vectors import l2_normalize, to_float32

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """What the rest of the app needs from an embedding model."""

    config: EmbeddingConfig
    contract: EmbeddingContract

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddedText]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


def model_dimension(model_name: str) -> int:
    """Look up the vector size of a model in FastEmbed's list (no download).

    Only models on FastEmbed's own supported list can be used, so a model
    name can never point at arbitrary files or code.
    """
    for description in TextEmbedding.list_supported_models():
        if description["model"] == model_name:
            return int(description["dim"])
    raise UnsupportedEmbeddingModelError()


def _check_texts(texts: Sequence[str]) -> None:
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")


class FastEmbedProvider:
    """One real local provider. One instance = one loaded model, reused."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.contract = EmbeddingContract(
            model_name=config.model_name,
            embedding_version=config.embedding_version,
            dimension=model_dimension(config.model_name),
            dtype=VECTOR_DTYPE,
            normalized=config.normalize_embeddings,
        )
        self._model: TextEmbedding | None = None
        self._load_lock = threading.Lock()  # two requests must not load it twice

    def _get_model(self) -> TextEmbedding:
        with self._load_lock:
            if self._model is None:
                try:
                    self._model = TextEmbedding(
                        model_name=self.config.model_name,
                        cache_dir=str(self.config.cache_dir),
                    )
                except Exception as error:  # download or model file problems
                    logger.error("Could not load the embedding model: %s", type(error).__name__)
                    raise EmbeddingModelUnavailableError()
                logger.info("Loaded embedding model %s", self.config.model_name)
            return self._model

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddedText]:
        """Embed several texts in one model call (they are one batch)."""
        texts = list(texts)
        if not texts:
            return []
        _check_texts(texts)
        model = self._get_model()
        try:
            raw_vectors = [array.tolist() for array in model.embed(texts, batch_size=len(texts))]
            truncated = [_is_truncated(model, text) for text in texts]
        except Exception as error:
            logger.error("Embedding model failed: %s", type(error).__name__)
            raise EmbeddingError()
        if len(raw_vectors) != len(texts):
            raise EmbeddingError()
        return [
            EmbeddedText(vector=self._finish(values), truncated=flag)
            for values, flag in zip(raw_vectors, truncated)
        ]

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a (future) search question with the SAME model and contract.

        For the default model FastEmbed adds no query prefix, so a query and a
        document with the same text get the same vector.
        """
        _check_texts([text])
        model = self._get_model()
        try:
            values = next(iter(model.query_embed(text))).tolist()
        except Exception as error:
            logger.error("Embedding model failed: %s", type(error).__name__)
            raise EmbeddingError()
        return self._finish(values)

    def _finish(self, values: list[float]) -> tuple[float, ...]:
        """Check the size, normalize if configured, round to float32."""
        if len(values) != self.contract.dimension:
            raise EmbeddingDimensionError()
        try:
            if self.contract.normalized:
                values = l2_normalize(values)
            return to_float32(values)
        except ValueError:  # NaN, infinity or an all-zero vector
            raise EmbeddingError()


def _is_truncated(model: Any, text: str) -> bool:
    """Whether the model cut the text because it has too many tokens.

    The default model reads at most 128 tokens (about 500 English characters
    or 250 Chinese characters); anything after that does not affect the
    vector. FastEmbed's tokenizer reports the cut-off part as "overflowing".
    (FastEmbed exposes its tokenizer as model.model.tokenizer; the version
    is pinned in requirements.txt.)
    """
    return bool(model.model.tokenizer.encode(text).overflowing)
