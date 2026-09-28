"""Batch 6B: compare embedding models on the frozen Batch 6A dataset (offline).

Run from the project root after the models were downloaded explicitly
(python -m app.evaluation.download_candidates <candidate>):

    .venv\\Scripts\\python.exe -m app.evaluation.compare_models
    .venv\\Scripts\\python.exe -m app.evaluation.compare_models --models e5-small,e5-base

It prints a Markdown report and writes nothing into the project: every
knowledge base lives in a temporary folder, the application's database and
stored vectors are never opened. Network access is switched off
(HF_HUB_OFFLINE=1, set before huggingface_hub is imported). A model that is
not cached is reported as skipped; it is never downloaded here.

Controlled experiment: only the embedding model changes. Chunking (1200/200),
search, ranking, top_k, metrics and the dataset are the Batch 6A ones.

Each model runs in its own fresh child process (same Python, same machine,
same settings: ONNX Runtime CPU with its default thread count, the
application's embedding batch size). A child's memory peak therefore belongs
to exactly one model. The child writes its ModelResult as JSON into a
temporary file; the parent validates it and renders the report.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # must be set before huggingface_hub is imported

import argparse  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections.abc import Callable, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

from app.document_processing.chunking import ChunkingConfig  # noqa: E402
from app.embeddings.provider import FastEmbedProvider, model_files  # noqa: E402
from app.embeddings.vectors import text_sha256  # noqa: E402
from app.evaluation import comparison_report  # noqa: E402
from app.evaluation.analysis import fingerprint  # noqa: E402
from app.evaluation.candidates import (  # noqa: E402
    BASELINE_KEY,
    CANDIDATES,
    CandidateModel,
    get_candidate,
    make_provider,
    missing_files,
    snapshot_folder,
    verify_official_files,
)
from app.evaluation.comparison import (  # noqa: E402
    DATASET_SHA256,
    EXPECTED_CHUNKS,
    IncompleteResultError,
    ModelResult,
    check_same_chunks,
    collect_result,
    validate_result,
)
from app.evaluation.golden import QUERIES  # noqa: E402
from app.evaluation.dataset_versions import VERSIONS, verify_dataset  # noqa: E402
from app.evaluation.resources import (  # noqa: E402
    SystemMemory,
    files_size,
    folder_size,
    process_memory,
    system_memory,
)
from app.evaluation.runner import KnowledgeBase, build_knowledge_base, evaluate  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE_DIR = PROJECT_ROOT / "storage" / "model_cache"
CHILD_TIMEOUT_SECONDS = 45 * 60
SAMPLE_CHUNKS = 3  # chunks embedded twice to check determinism
NORM_TOLERANCE = 1e-5  # float32 rounding of a length-1 vector
EXIT_BASELINE_MISSING, EXIT_DATASET_CHANGED, EXIT_CHILD_FAILED = 1, 3, 4
KNOWN_PEAK_GIB = {"e5-small": 1.67, "e5-base": 2.28, "bge-m3": 3.4}


def comparison_dataset_sha(dataset_version: str, artifact_sha: str) -> str:
    # Keep the exact Batch 6B fingerprint contract for retrieval_eval_v1.
    return DATASET_SHA256 if dataset_version == "retrieval_eval_v1" else artifact_sha


def selected_model_keys(models: Sequence[str], *, skip_bge: bool = False,
                        only_bge: bool = False) -> list[str]:
    keys = ["bge-m3"] if only_bge else list(models)
    return [key for key in keys if not (skip_bge and key == "bge-m3")]


def resource_preflight(candidate: CandidateModel, memory: SystemMemory) -> str | None:
    """Require known peak plus 25% reserve in free physical and commit memory."""
    peak = KNOWN_PEAK_GIB.get(candidate.key)
    if peak is None:
        return None
    required_gib = peak * 1.25
    if (memory.available_physical is None or memory.available_commit is None
            or memory.available_physical < required_gib * 1024**3
            or memory.available_commit < required_gib * 1024**3):
        available_ram = ("unavailable" if memory.available_physical is None else
                         f"{memory.available_physical / 1024**3:.2f} GiB")
        available_commit = ("unavailable" if memory.available_commit is None else
                            f"{memory.available_commit / 1024**3:.2f} GiB")
        return (f"NOT MEASURED — insufficient safe memory (requires about {required_gib:.2f} GiB; "
                f"available {available_ram} RAM and {available_commit} commit, against the "
                f"prior {peak:.2f} GiB peak plus 25% reserve).")
    return None


class CandidateNotCachedError(RuntimeError):
    pass


class ModelFilesMismatchError(RuntimeError):
    pass


# --- One model (runs inside its own child process) ------------------------------


def record_model_inputs(model: object) -> list[str]:
    """Record every text handed to model.embed (the exact model input, prefix included).

    Wraps the method of this one loaded object only; nothing is logged.
    """
    seen: list[str] = []
    original = model.embed

    def recording_embed(texts, batch_size):
        texts = list(texts)
        seen.extend(texts)
        return original(texts, batch_size=batch_size)

    model.embed = recording_embed
    return seen


def input_checks(provider: FastEmbedProvider, kb: KnowledgeBase,
                 seen: Sequence[str], queries=QUERIES) -> dict[str, bool]:
    """Did the model receive exactly the documented inputs?

    Every chunk as passage_prefix + text, every question as query_prefix +
    text, and nothing else (no swapped, missing or extra prefix).
    """
    passages = {provider.contract.passage_input(chunk.text) for chunk in kb.chunks.values()}
    query_inputs = {provider.contract.query_input(query.text) for query in queries}
    return {
        "passage_inputs_follow_contract": passages <= set(seen),
        "query_inputs_follow_contract": query_inputs <= set(seen),
        "no_other_model_inputs": set(seen) <= passages | query_inputs,
    }


def vector_checks(provider: FastEmbedProvider, kb: KnowledgeBase, queries=QUERIES) -> dict[str, bool]:
    """Dimension, length 1, finite values, determinism and the input hash of real vectors."""
    texts = [chunk.text for chunk in list(kb.chunks.values())[:SAMPLE_CHUNKS]]
    first, second = provider.embed_documents(texts), provider.embed_documents(texts)
    query_vector = provider.embed_query(queries[0].text)
    vectors = [item.vector for item in first] + [query_vector]
    dimension = provider.contract.dimension
    return {
        "dimension": all(len(vector) == dimension for vector in vectors),
        "finite": all(math.isfinite(value) for vector in vectors for value in vector),
        "normalized": all(abs(math.sqrt(math.fsum(v * v for v in vector)) - 1.0) < NORM_TOLERANCE
                          for vector in vectors),
        "deterministic_vectors": [i.vector for i in first] == [i.vector for i in second],
        "input_hash_is_prefixed_passage": all(
            item.input_sha256 == text_sha256(provider.contract.passage_input(text))
            for item, text in zip(first, texts)),
    }


def run_one(candidate: CandidateModel, cache_dir: Path,
            dataset_version: str = "retrieval_eval_v1") -> ModelResult:
    """Evaluate one cached model on one checked dataset. Never downloads."""
    dataset = verify_dataset(dataset_version)
    missing = missing_files(candidate, cache_dir)
    if missing:
        raise CandidateNotCachedError(f"{candidate.key}: {len(missing)} files not cached")
    folder = snapshot_folder(candidate.spec, cache_dir)
    problems = verify_official_files(candidate, folder)
    if problems:
        raise ModelFilesMismatchError("; ".join(problems))
    before = process_memory()
    provider = make_provider(candidate, cache_dir)
    started = time.perf_counter()
    model = provider._get_model()  # load now, so that loading is timed on its own
    load_seconds = time.perf_counter() - started
    after_load = process_memory()
    seen = record_model_inputs(model)

    with tempfile.TemporaryDirectory(prefix="docubot-compare-") as root:
        kb = build_knowledge_base(Path(root), provider, ChunkingConfig(), dataset.passages)
        results = evaluate(kb, provider, dataset.queries)
        repeat = evaluate(kb, provider, dataset.queries)  # queries embedded and searched again
    checks = {
        "official_files_match_spec": True,  # verified above, would have raised
        "revision_pinned": (folder.name == candidate.spec.revision
                            == provider.contract.model_revision and len(folder.name) == 40),
        "tokenizer_limit": model.model.tokenizer.truncation["max_length"]
                           == candidate.spec.max_tokens,
        "deterministic_ranking": fingerprint(results) == fingerprint(repeat),
        "cpu_only": model.model.model.get_providers() == ["CPUExecutionProvider"],
    }
    checks |= input_checks(provider, kb, seen, dataset.queries)
    checks |= vector_checks(provider, kb, dataset.queries)
    end = process_memory()
    searches = [result.search_seconds * 1000 for result in results]
    weights = [candidate.spec.model_file, *candidate.spec.external_data_files]
    resources = {
        "download_bytes": files_size(folder, model_files(candidate.spec)),
        "weights_bytes": files_size(folder, weights),
        "cache_bytes": folder_size(folder.parents[1]),
        "load_seconds": load_seconds,
        "embed_seconds": kb.embed_seconds,
        "search_ms_mean": sum(searches) / len(searches),
        "search_ms_max": max(searches),
        "private_bytes_before_load": before.private_bytes,
        "private_bytes_after_load": after_load.private_bytes,
        "peak_working_set": end.peak_working_set,
        "peak_private_bytes": end.peak_private_bytes,
        "embedding_batch_size": provider.config.batch_size,
    }
    return collect_result(candidate.key, candidate.spec, kb, results, checks=checks,
                          resources=resources, dataset_version=dataset.version,
                          dataset_sha256=comparison_dataset_sha(dataset.version, dataset.sha256),
                          hard_negatives=dataset.hard_negatives)


def child_main(key: str, json_path: Path, dataset_version: str) -> int:
    logging.getLogger("app").setLevel(logging.WARNING)
    try:
        dataset = verify_dataset(dataset_version)
        result = run_one(get_candidate(key), MODEL_CACHE_DIR, dataset_version)
        validate_result(result, expected_queries=dataset.queries,
                        expected_passages=dataset.passages,
                        expected_sha256=comparison_dataset_sha(dataset.version, dataset.sha256),
                        expected_chunk_count=None if dataset_version == "retrieval_eval_v2" else EXPECTED_CHUNKS)
    except Exception as error:  # reported by the parent as "failed", with the type only
        print(f"{key}: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_CHILD_FAILED
    json_path.write_text(result.to_json(), encoding="utf-8")
    return 0


# --- The parent: one child per model, then the report ---------------------------


def estimate_peak_bytes(candidate: CandidateModel, completed: Sequence[ModelResult]) -> int | None:
    """Expected peak private memory of a model, from the models MEASURED in this run.

    A straight line through (download size, measured peak) of the two
    largest completed models: fixed overhead (Python, tokenizer, SQLite)
    plus a part that grows with the weights. None if fewer than two points.
    """
    points = sorted((r.resources["download_bytes"], r.resources["peak_private_bytes"])
                    for r in completed if r.resources.get("peak_private_bytes"))
    if len(points) < 2:
        return None
    (x1, y1), (x2, y2) = points[-2], points[-1]
    if x2 == x1:
        return None
    slope = (y2 - y1) / (x2 - x1)
    return int(y2 + slope * (candidate.download_bytes - x2))


def resource_skip_reason(candidate: CandidateModel, completed: Sequence[ModelResult],
                         memory: SystemMemory) -> str | None:
    """Why an optional model must not be run on this laptop now, or None."""
    estimate = estimate_peak_bytes(candidate, completed)
    if estimate is None or memory.available_physical is None:
        return None
    gb = 1024**3
    if estimate > memory.available_physical or (
            memory.available_commit is not None and estimate > memory.available_commit):
        return (f"Candidate B skipped due to measured resource constraints: estimated peak "
                f"{estimate / gb:.1f} GB (extrapolated from the measured E5 runs) > available "
                f"RAM {memory.available_physical / gb:.1f} GB (commit headroom "
                f"{(memory.available_commit or 0) / gb:.1f} GB) at the time of the run.")
    return None


def run_child(key: str, timeout: float, dataset_version: str = "retrieval_eval_v1") -> tuple[ModelResult | None, str | None]:
    with tempfile.TemporaryDirectory(prefix="docubot-compare-result-") as folder:
        json_path = Path(folder) / f"{key}.json"
        command = [sys.executable, "-m", "app.evaluation.compare_models", "--run-one", key,
                   "--dataset-version", dataset_version,
                   "--json", str(json_path)]
        try:
            finished = subprocess.run(command, cwd=PROJECT_ROOT, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return None, f"stopped after {timeout / 60:.0f} minutes (too slow on this laptop)"
        if finished.returncode != 0 or not json_path.is_file():
            return None, f"failed in its own process (exit code {finished.returncode})"
        return ModelResult.from_json(json_path.read_text(encoding="utf-8")), None


def environment() -> dict[str, str]:
    import fastembed
    import onnxruntime

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "logical_cpus": str(os.cpu_count()),
        "onnxruntime": onnxruntime.__version__,
        "fastembed": fastembed.__version__,
        "threads": "ONNX Runtime default (not set), identical for every model",
    }


def compare(keys: Sequence[str], timeout: float = CHILD_TIMEOUT_SECONDS,
            run: Callable[[str, float], tuple[ModelResult | None, str | None]] = run_child,
            memory: Callable[[], SystemMemory] = system_memory,
            cache_dir: Path = MODEL_CACHE_DIR,
            dataset_version: str = "retrieval_eval_v1",
            allow_only_candidate: bool = False) -> tuple[list[ModelResult], dict[str, str]]:
    """Run every requested model (baseline first); return the results and the skip reasons."""
    candidates = [get_candidate(key) for key in keys]  # the allowlist
    if not candidates or (not allow_only_candidate and candidates[0].key != BASELINE_KEY):
        raise ValueError(f"The first model must be the baseline {BASELINE_KEY!r}.")
    dataset = verify_dataset(dataset_version)
    results: list[ModelResult] = []
    skipped: dict[str, str] = {}
    for candidate in candidates:
        missing = missing_files(candidate, cache_dir)
        if missing:
            skipped[candidate.key] = (f"not cached ({len(missing)} files missing); download it "
                                      f"explicitly: python -m app.evaluation.download_candidates "
                                      f"{candidate.key}")
            continue
        if run is run_child:
            reason = resource_preflight(candidate, memory())
            if reason:
                skipped[candidate.key] = reason
                continue
        if candidate.optional and run is run_child and len(results) >= 2:
            reason = resource_skip_reason(candidate, results, memory())
            if reason:
                skipped[candidate.key] = reason
                continue
        if run is run_child:
            result, error = run_child(candidate.key, timeout, dataset_version)
        else:
            result, error = run(candidate.key, timeout)
        if result is None:
            skipped[candidate.key] = error or "failed"
            continue
        validate_result(result, expected_queries=dataset.queries,
                        expected_passages=dataset.passages,
                        expected_sha256=comparison_dataset_sha(dataset.version, dataset.sha256),
                        expected_chunk_count=None if dataset_version == "retrieval_eval_v2" else EXPECTED_CHUNKS)
        results.append(result)
    check_same_chunks(results)
    return results, skipped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default=",".join(CANDIDATES),
                        help="comma-separated candidate keys, baseline first")
    parser.add_argument("--dataset-version", choices=VERSIONS, default="retrieval_eval_v1")
    parser.add_argument("--skip-bge", action="store_true", help="do not attempt BGE-M3")
    parser.add_argument("--only-bge", action="store_true", help="evaluate only BGE-M3 if safe and cached")
    parser.add_argument("--report", type=Path,
                        help="also save the Markdown report to this path")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    parser.add_argument("--json", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.run_one:
        return child_main(args.run_one, args.json, args.dataset_version)

    try:
        dataset = verify_dataset(args.dataset_version)
    except (OSError, ValueError, KeyError) as error:
        print(f"Dataset verification failed: {error}", file=sys.stderr)
        return EXIT_DATASET_CHANGED
    started = time.perf_counter()
    keys = selected_model_keys([key.strip() for key in args.models.split(",") if key.strip()],
                               skip_bge=args.skip_bge, only_bge=args.only_bge)
    try:
        results, skipped = compare(keys, dataset_version=args.dataset_version,
                                   allow_only_candidate=args.only_bge)
    except (ValueError, IncompleteResultError) as error:
        print(error, file=sys.stderr)
        return EXIT_CHILD_FAILED
    if (not results and skipped and all(
            reason.startswith("NOT MEASURED") or
            (args.only_bge and reason.startswith("not cached"))
            for reason in skipped.values())):
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        report = (f"Dataset: {dataset.version} ({dataset.sha256})\n\n"
                  "No candidate model was measured; see the skip reasons below.\n\n"
                  + "\n".join(f"- {key}: {reason}" for key, reason in skipped.items()))
        if args.report:
            args.report.write_text(report + "\n", encoding="utf-8")
        print(report)
        return 0
    if not results or (not args.only_bge and results[0].key != BASELINE_KEY):
        print(f"The baseline could not be evaluated: {skipped.get(BASELINE_KEY)}", file=sys.stderr)
        return EXIT_BASELINE_MISSING
    # A redirected stdout uses the Windows locale encoding (e.g. GBK); the report is UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")
    report = (f"Dataset: {dataset.version} ({dataset.sha256})\n\n" +
              comparison_report.render(results, skipped, environment(), time.perf_counter() - started))
    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
