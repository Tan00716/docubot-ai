"""The local embedding model (FastEmbed + ONNX Runtime on the CPU).

This is the ONLY module that imports fastembed, tokenizers or huggingface_hub.
Everything outside it works with plain Python tuples of floats, so the model
can be replaced later without touching storage or the API.

Lifecycle: creating a FastEmbedProvider is cheap. The model is loaded the
first time text is embedded, then reused for every later call (never once
per chunk). On the very first load the model files of the pinned revision
are downloaded into config.cache_dir; after that, loading uses only the
local files. Embedding itself always runs locally: no hosted API is called.

Input contract (E5 models): the model must be told what kind of text it is
reading. Document chunks are embedded as "passage: <text>" and questions as
"query: <text>". The prefix is added here, at embedding time only.
"""

import json
import logging
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from app.embeddings.config import (
    VECTOR_DTYPE,
    EmbeddingConfig,
    EmbeddingModelSpec,
    get_model_spec,
)
from app.embeddings.models import (
    EmbeddedText,
    EmbeddingContract,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    EmbeddingModelUnavailableError,
    QueryTooLongError,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.vectors import l2_normalize, text_sha256, to_float32

logger = logging.getLogger(__name__)

# The only files downloaded for a model: its tokenizer, its configuration and
# the ONNX weights. No Python code and no pickle files are ever fetched.
TOKENIZER_AND_CONFIG_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

POOLING_TYPES = {"mean": PoolingType.MEAN}

# Models that this process has registered with FastEmbed, with the spec used
# (see _register_model). FastEmbed cannot re-register a name, so a changed
# spec for the same name must fail loudly instead of silently using the old one.
_registered_models: dict[str, EmbeddingModelSpec] = {}
_registration_lock = threading.Lock()


class EmbeddingProvider(Protocol):
    """What the rest of the app needs from an embedding model."""

    config: EmbeddingConfig
    contract: EmbeddingContract

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddedText]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


def build_contract(config: EmbeddingConfig) -> EmbeddingContract:
    """The contract for a config (no download). Unknown models are rejected."""
    spec = get_model_spec(config.model_name)
    return EmbeddingContract(
        model_name=spec.name,
        model_revision=spec.revision,
        embedding_version=config.embedding_version,
        dimension=spec.dimension,
        max_tokens=spec.max_tokens,
        passage_prefix=spec.passage_prefix,
        query_prefix=spec.query_prefix,
        dtype=VECTOR_DTYPE,
        normalized=config.normalize_embeddings,
    )


def model_files(spec: EmbeddingModelSpec) -> list[str]:
    return [*TOKENIZER_AND_CONFIG_FILES, spec.model_file]


def download_model_files(spec: EmbeddingModelSpec, cache_dir: Path) -> Path:
    """Folder with the model files of exactly spec.revision (downloaded once).

    The local copy is used whenever it is complete, so no network is needed
    after the first run. huggingface_hub keeps the files in cache_dir under
    models--<org>--<name>/snapshots/<revision>/.
    """
    options: dict[str, Any] = {
        "repo_id": spec.name,
        "revision": spec.revision,
        "allow_patterns": model_files(spec),
        "cache_dir": str(cache_dir),
    }
    try:
        folder = Path(snapshot_download(**options, local_files_only=True))
        if all((folder / name).is_file() for name in model_files(spec)):
            return folder
    except Exception:  # not in the cache yet
        pass
    return Path(snapshot_download(**options))


def _register_model(spec: EmbeddingModelSpec) -> None:
    """Tell FastEmbed how to run a model that is not on its built-in list.

    Uses FastEmbed's documented TextEmbedding.add_custom_model(). FastEmbed's
    own normalization is switched off: DocuBot normalizes in _finish(), so the
    contract's "normalized" field is always the truth.
    """
    with _registration_lock:
        registered = _registered_models.get(spec.name)
        if registered == spec:
            return
        if registered is not None:
            logger.error("%s was registered with another spec; restart the process", spec.name)
            raise UnsupportedEmbeddingModelError()
        built_in = {d["model"].lower() for d in TextEmbedding.list_supported_models()}
        if spec.name.lower() in built_in:
            # FastEmbed would use its own settings and silently ignore ours.
            logger.error("FastEmbed already defines %s; refusing to guess its settings", spec.name)
            raise UnsupportedEmbeddingModelError()
        TextEmbedding.add_custom_model(
            model=spec.name,
            pooling=POOLING_TYPES[spec.pooling],
            normalization=False,
            sources=ModelSource(hf=spec.name),
            dim=spec.dimension,
            model_file=spec.model_file,
            license=spec.license,
        )
        _registered_models[spec.name] = spec


def check_model_metadata(model: Any, model_dir: Path, spec: EmbeddingModelSpec) -> None:
    """Refuse model files whose real settings differ from the spec.

    Checks the vector size in config.json and the token limit that the
    loaded tokenizer really applies. (FastEmbed exposes its tokenizer as
    model.model.tokenizer; the version is pinned in requirements.txt.)
    """
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        hidden_size = config.get("hidden_size")
        truncation = model.model.tokenizer.truncation or {}
        max_tokens = truncation.get("max_length")
    except (OSError, ValueError, AttributeError) as error:
        logger.error("Could not read the embedding model metadata: %s", type(error).__name__)
        raise EmbeddingModelMismatchError()
    if hidden_size != spec.dimension or max_tokens != spec.max_tokens:
        logger.error(
            "Model files of %s do not match the spec: dimension %r (expected %d), "
            "max tokens %r (expected %d)",
            spec.name, hidden_size, spec.dimension, max_tokens, spec.max_tokens,
        )
        raise EmbeddingModelMismatchError()


def _check_texts(texts: Sequence[str]) -> None:
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")


class FastEmbedProvider:
    """One real local provider. One instance = one loaded model, reused."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.spec = get_model_spec(config.model_name)
        self.contract = build_contract(config)
        self._model: TextEmbedding | None = None
        self._counting_tokenizer: Tokenizer | None = None  # see count_tokens()
        self._load_lock = threading.Lock()  # two requests must not load it twice

    def _get_model(self) -> TextEmbedding:
        with self._load_lock:
            return self._get_model_unlocked()

    def _get_model_unlocked(self) -> TextEmbedding:
        # The caller must hold self._load_lock (threading.Lock is not re-entrant).
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> TextEmbedding:
        _register_model(self.spec)
        try:
            model_dir = download_model_files(self.spec, self.config.cache_dir)
            model = TextEmbedding(
                model_name=self.spec.name,
                cache_dir=str(self.config.cache_dir),
                specific_model_path=str(model_dir),
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:  # download or model file problems
            logger.error("Could not load the embedding model: %s", type(error).__name__)
            raise EmbeddingModelUnavailableError()
        check_model_metadata(model, model_dir, self.spec)
        logger.info("Loaded embedding model %s@%s", self.spec.name, self.spec.revision[:12])
        return model

    def embed_documents(self, texts: Sequence[str]) -> list[EmbeddedText]:
        """Embed document chunks ("passage: " + text) in one model call."""
        texts = list(texts)
        if not texts:
            return []
        _check_texts(texts)
        inputs = [self.contract.passage_input(text) for text in texts]
        model = self._get_model()
        try:
            raw_vectors = [array.tolist() for array in model.embed(inputs, batch_size=len(inputs))]
            truncated = [_is_truncated(model, model_input) for model_input in inputs]
        except Exception as error:
            logger.error("Embedding model failed: %s", type(error).__name__)
            raise EmbeddingError()
        if len(raw_vectors) != len(inputs):
            raise EmbeddingError()
        return [
            EmbeddedText(vector=self._finish(values), truncated=flag,
                         input_sha256=text_sha256(model_input))
            for values, flag, model_input in zip(raw_vectors, truncated, inputs)
        ]

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a search question ("query: " + text), same model and contract.

        model.embed() is used for both kinds of text, so the only prefix is
        the one added here (FastEmbed adds none for custom models).

        Token limit policy: the question is measured with the model's own
        (already loaded) tokenizer. Up to max_tokens tokens, prefix and
        special tokens included, it is embedded; one token more raises
        QueryTooLongError. The model would otherwise silently ignore the rest.
        """
        _check_texts([text])
        model = self._get_model()
        model_input = self.contract.query_input(text)
        try:
            too_long = _is_truncated(model, model_input)
        except Exception as error:
            logger.error("Tokenizer failed: %s", type(error).__name__)
            raise EmbeddingError()
        if too_long:
            # Only the fact is logged, never the question itself.
            logger.info("Rejected a search query longer than %d tokens", self.contract.max_tokens)
            raise QueryTooLongError()
        try:
            values = next(iter(model.embed([model_input], batch_size=1)))
            values = values.tolist()
        except Exception as error:
            logger.error("Embedding model failed: %s", type(error).__name__)
            raise EmbeddingError()
        return self._finish(values)

    def count_tokens(self, model_input: str) -> int:
        """How many tokens the model input has BEFORE the max_tokens cut (for analysis).

        model_input must already contain its prefix ("passage: " or "query: ").
        The loaded tokenizer cuts every input at max_tokens, so it cannot say
        how long a text really is. This uses a copy of that same tokenizer
        with the cut switched off (made once from the loaded model, the model
        itself is not loaded again). It never changes what gets embedded.
        """
        with self._load_lock:
            if self._counting_tokenizer is None:
                self._counting_tokenizer = _untruncated_copy(self._get_model_unlocked())
            tokenizer = self._counting_tokenizer
        return len(tokenizer.encode(model_input).ids)

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


def _is_truncated(model: Any, model_input: str) -> bool:
    """Whether the model cut the input because it has too many tokens.

    multilingual-e5-small reads at most 512 tokens (prefix included);
    anything after that does not affect the vector. FastEmbed's tokenizer
    reports the cut-off part as "overflowing".
    """
    return bool(model.model.tokenizer.encode(model_input).overflowing)


def _untruncated_copy(model: Any) -> Tokenizer:
    """An independent copy of the model's tokenizer that never cuts or pads.

    Built from the loaded tokenizer's own JSON, so the vocabulary and rules
    are exactly the ones the model uses. The original is left untouched.
    """
    copy = Tokenizer.from_str(model.model.tokenizer.to_str())
    copy.no_truncation()
    copy.no_padding()
    return copy
