"""Candidate embedding models for the offline model comparison (Batch 6B).

EVALUATION ONLY. The candidate specs below are deliberately NOT in
app.embeddings.config.SUPPORTED_MODELS, so the application (API, bot,
stored vectors) cannot be configured with them. They are only used by
app/evaluation/compare_models.py, which builds throw-away knowledge bases
in temporary folders.

Every value was copied from the model's official Hugging Face files at the
pinned revision (checked 2026-09-28), and verify_official_files() re-checks
them against the downloaded files before a model is evaluated:

    config.json            hidden_size         -> dimension
    tokenizer_config.json  model_max_length    -> max_tokens
    1_Pooling/config.json  pooling_mode_*      -> pooling
    modules.json           a Normalize module  -> vectors must be normalized

Prefixes are model-specific (never an E5 default for every model):
    multilingual-e5-*  model card FAQ: "query: " / "passage: " are required,
                       "otherwise you will see a performance degradation"
    BAAI/bge-m3        model card FAQ: for dense retrieval BGE-M3 "no longer
                       requires adding instructions to the queries"; passages
                       never had one -> both prefixes are empty
"""

import json
from dataclasses import dataclass
from pathlib import Path

from app.embeddings.config import MULTILINGUAL_E5_SMALL, EmbeddingConfig, EmbeddingModelSpec
from app.embeddings.provider import FastEmbedProvider, model_files

# Read only to double-check the spec; nothing from the repository is executed.
METADATA_FILES = ("1_Pooling/config.json", "modules.json")
NORMALIZE_MODULE = "sentence_transformers.models.Normalize"
POOLING_KEYS = {"mean": "pooling_mode_mean_tokens", "cls": "pooling_mode_cls_token"}


class UnknownCandidateError(ValueError):
    """The name is not on the candidate allowlist."""


@dataclass(frozen=True)
class CandidateModel:
    key: str  # short name used on the command line
    spec: EmbeddingModelSpec
    languages: str  # as stated by the model card
    runtime: str
    is_production: bool  # True only for the model the application uses today
    # Total size of exactly the files model_files() downloads, from the Hub's
    # file metadata at the pinned revision (the cold download size).
    download_bytes: int
    optional: bool = False  # evaluated only if this laptop can run it (measured)


E5_SMALL = CandidateModel(
    key="e5-small",
    spec=MULTILINGUAL_E5_SMALL,  # the production spec object itself, unchanged
    languages="94 (model card metadata)",
    runtime="FastEmbed custom model, ONNX Runtime CPU, float32",
    is_production=True,
    download_bytes=487_352_505,
)

E5_BASE = CandidateModel(
    key="e5-base",
    spec=EmbeddingModelSpec(
        name="intfloat/multilingual-e5-base",
        revision="d128750597153bb5987e10b1c3493a34e5a4502a",
        model_file="onnx/model.onnx",  # full-precision float32 ONNX (about 1.1 GB)
        dimension=768,
        max_tokens=512,
        pooling="mean",
        passage_prefix="passage: ",
        query_prefix="query: ",
        license="mit",
        metadata_files=METADATA_FILES,
    ),
    languages="94 (model card metadata)",
    runtime="FastEmbed custom model, ONNX Runtime CPU, float32",
    is_production=False,
    download_bytes=1_127_143_723,
)

BGE_M3 = CandidateModel(
    key="bge-m3",
    spec=EmbeddingModelSpec(
        name="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        model_file="onnx/model.onnx",  # graph only; the weights are external data
        # about 2.27 GB of float32 weights next to the graph
        external_data_files=("onnx/model.onnx_data", "onnx/Constant_7_attr__value"),
        dimension=1024,
        max_tokens=8192,
        pooling="cls",
        passage_prefix="",
        query_prefix="",
        license="mit",
        metadata_files=METADATA_FILES,
    ),
    languages="more than 100 (model card)",
    runtime="FastEmbed custom model, ONNX Runtime CPU, float32, dense vectors only",
    is_production=False,
    download_bytes=2_284_711_826,
    optional=True,
)

# The allowlist, in report order. The first entry is the baseline.
CANDIDATES: dict[str, CandidateModel] = {c.key: c for c in (E5_SMALL, E5_BASE, BGE_M3)}
BASELINE_KEY = E5_SMALL.key


def get_candidate(key: str) -> CandidateModel:
    candidate = CANDIDATES.get(key)
    if candidate is None:
        raise UnknownCandidateError(f"Unknown candidate {key!r}; allowed: {', '.join(CANDIDATES)}.")
    return candidate


def make_provider(candidate: CandidateModel, cache_dir: Path) -> FastEmbedProvider:
    """The same FastEmbedProvider the application uses.

    The production model goes through the unchanged production path (no
    explicit spec). A candidate passes its spec explicitly, because the
    application's allowlist does not know it.
    """
    config = EmbeddingConfig(cache_dir=cache_dir, model_name=candidate.spec.name)
    if candidate.is_production:
        return FastEmbedProvider(config)
    return FastEmbedProvider(config, spec=candidate.spec)


def snapshot_folder(spec: EmbeddingModelSpec, cache_dir: Path) -> Path:
    """Where huggingface_hub keeps exactly this revision: one folder per repo AND commit."""
    return cache_dir / ("models--" + spec.name.replace("/", "--")) / "snapshots" / spec.revision


def missing_files(candidate: CandidateModel, cache_dir: Path) -> list[str]:
    """Files of the pinned revision that are not in the local cache."""
    folder = snapshot_folder(candidate.spec, cache_dir)
    return [name for name in model_files(candidate.spec) if not (folder / name).is_file()]


def _read_json(folder: Path, name: str) -> object:
    return json.loads((folder / name).read_text(encoding="utf-8"))


def verify_official_files(candidate: CandidateModel, folder: Path) -> list[str]:
    """Differences between the spec and the model's own files (empty = all match)."""
    spec = candidate.spec
    problems = []
    config = _read_json(folder, "config.json")
    if config.get("hidden_size") != spec.dimension:
        problems.append(f"config.json hidden_size {config.get('hidden_size')!r} "
                        f"!= {spec.dimension}")
    tokenizer_config = _read_json(folder, "tokenizer_config.json")
    if tokenizer_config.get("model_max_length") != spec.max_tokens:
        problems.append(f"tokenizer_config.json model_max_length "
                        f"{tokenizer_config.get('model_max_length')!r} != {spec.max_tokens}")
    if not spec.metadata_files:
        return problems  # the production spec was checked when it was added (Batch 5A)
    pooling = _read_json(folder, "1_Pooling/config.json")
    active = [mode for mode, key in POOLING_KEYS.items() if pooling.get(key) is True]
    if active != [spec.pooling]:
        problems.append(f"1_Pooling/config.json pooling {active!r} != [{spec.pooling!r}]")
    if pooling.get("word_embedding_dimension") != spec.dimension:
        problems.append("1_Pooling/config.json word_embedding_dimension != dimension")
    modules = _read_json(folder, "modules.json")
    if not any(module.get("type") == NORMALIZE_MODULE for module in modules):
        problems.append("modules.json has no Normalize module")
    return problems
