"""Retrieval evaluation report (development only, not a general benchmark).

Run from the project root once the model has been downloaded (any earlier
embedding run does that):

    .venv\\Scripts\\python.exe -m app.evaluation.report

It prints a Markdown report to the terminal and writes nothing else:
every knowledge base is built in a temporary folder that is deleted
afterwards, the project database is never touched. Network access is
switched off (HF_HUB_OFFLINE=1); if the pinned model is not cached yet the
report stops instead of downloading it.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"  # must be set before huggingface_hub is imported

import logging  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections.abc import Sequence  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

from app.document_processing.chunking import ChunkingConfig  # noqa: E402
from app.embeddings.config import EmbeddingConfig, EmbeddingModelSpec, get_model_spec  # noqa: E402
from app.embeddings.provider import FastEmbedProvider  # noqa: E402
from app.evaluation import analysis  # noqa: E402
from app.evaluation.corpus import LANGUAGES, PASSAGES  # noqa: E402
from app.evaluation.golden import DIRECTIONS, QUERIES, validate_dataset  # noqa: E402
from app.evaluation.metrics import (  # noqa: E402
    MetricSummary,
    QueryOutcome,
    summarize,
    summarize_groups,
)
from app.evaluation.runner import (  # noqa: E402
    NOT_TRUNCATED,
    TRUNCATION_GROUPS,
    KnowledgeBase,
    QueryResult,
    TokenCountingProvider,
    build_knowledge_base,
    evaluate,
)

MODEL_CACHE_DIR = Path(__file__).resolve().parents[2] / "storage" / "model_cache"
PRODUCTION = ChunkingConfig()  # the Batch 4 defaults: 1200 / 200
EXPERIMENT_CONFIGS = (PRODUCTION, ChunkingConfig(800, 100), ChunkingConfig(600, 100),
                      ChunkingConfig(400, 50))
SHORT_PASSAGE, LONG_PASSAGE = 300, 700  # characters, only for describing the corpus
SAME_LANGUAGE_DIRECTIONS = ("en->en", "zh->zh")


@dataclass(frozen=True)
class ConfigRun:
    kb: KnowledgeBase  # its temporary database is gone; the chunk data stays in memory
    results: tuple[QueryResult, ...]
    seconds: float


def model_is_cached(spec: EmbeddingModelSpec, cache_dir: Path) -> bool:
    folder = "models--" + spec.name.replace("/", "--")
    return (cache_dir / folder / "snapshots" / spec.revision / spec.model_file).is_file()


def run_config(provider: TokenCountingProvider, config: ChunkingConfig) -> ConfigRun:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="docubot-eval-") as folder:
        kb = build_knowledge_base(Path(folder), provider, config)
        results = evaluate(kb, provider)
    return ConfigRun(kb=kb, results=results, seconds=time.perf_counter() - started)


# --- Formatting ---------------------------------------------------------------


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def metric_cells(summary: MetricSummary | None) -> list[str]:
    if summary is None:
        return ["0", "-", "-", "-", "-"]
    return [str(summary.queries), *(f"{summary.recall[k]:.3f}" for k in (1, 3, 5)),
            f"{summary.mrr:.3f}"]


def score_text(score: float | None) -> str:
    return "-" if score is None else f"{score:.4f}"


# --- Report sections ----------------------------------------------------------


def dataset_section(production: ConfigRun) -> str:
    rows = []
    for language in LANGUAGES:
        texts = [p.text for p in PASSAGES if p.language == language]
        rows.append([language, len(texts),
                     sum(len(t) < SHORT_PASSAGE for t in texts),
                     sum(SHORT_PASSAGE <= len(t) < LONG_PASSAGE for t in texts),
                     sum(len(t) >= LONG_PASSAGE for t in texts)])
    near = sum(450 <= c.tokens <= 512 for c in production.kb.chunks.values())
    return (f"Passages: {len(PASSAGES)} (one document each). Chunks at 1200/200: "
            f"{len(production.kb.chunks)} ({near} between 450 and 512 tokens). "
            f"Queries: {len(QUERIES)}.\n\n"
            + table(["Language", "Passages", f"short (<{SHORT_PASSAGE})",
                     "medium", f"long (>={LONG_PASSAGE} chars)"], rows))


def metrics_section(production: ConfigRun) -> str:
    outcomes = [r.outcome for r in production.results]
    by_direction = summarize_groups(outcomes, lambda o: o.direction, DIRECTIONS)
    rows = [["**all**", *metric_cells(summarize(outcomes))]]
    rows += [[direction, *metric_cells(summary)] for direction, summary in by_direction.items()]
    return table(["Direction", "Queries", "R@1", "R@3", "R@5", "MRR"], rows)


def query_section(production: ConfigRun) -> str:
    rows = []
    for result in production.results:
        d = analysis.diagnose(production.kb, result)
        secondary = ",".join("-" if r is None else str(r) for r in result.secondary_ranks)
        rows.append([d.query_id, d.direction, d.first_relevant_rank or "-",
                     score_text(d.first_relevant_score), d.competitor_language or "-",
                     score_text(d.competitor_score), secondary or "",
                     d.query_language_in_top, result.truncation_group])
    return table(["Query", "Direction", "Rank", "Score", "Top other chunk", "Its score",
                  "Secondary rank", f"Top-{analysis.TOP_N_FOR_LANGUAGE_MIX} in query language",
                  "Evidence"], rows)


def token_section(provider: TokenCountingProvider, production: ConfigRun) -> str:
    rows = [[s.language, s.chars, s.tokens, "yes" if s.truncated else "no"]
            for s in analysis.token_slices(provider)]
    limits = []
    for language in LANGUAGES:
        text = analysis.language_text(language)
        tokens = provider.count_tokens(provider.contract.passage_input(text))
        chunks = [c for c in production.kb.chunks.values() if c.language == language]
        limits.append([language, f"{len(text) / (tokens - 2):.2f}",
                       analysis.max_chars_within_limit(provider, text),
                       f"{sum(c.truncated for c in chunks)}/{len(chunks)}",
                       max(c.tokens for c in chunks)])
    return ("First N characters of each language's corpus text (\"passage: \" prefix and "
            "special tokens included):\n\n"
            + table(["Language", "Characters", "Tokens", "Over 512?"], rows)
            + "\n\n" + table(["Language", "Chars per token", "Max chars read completely",
                              "Truncated chunks at 1200/200", "Max chunk tokens"], limits))


def truncation_groups(run: ConfigRun) -> dict[str, str]:
    """query_id -> truncation group of its relevant evidence in this run."""
    return {r.query.query_id: r.truncation_group for r in run.results}


def _truncation_table(outcomes: Sequence[QueryOutcome], group_of: dict[str, str]) -> str:
    groups = summarize_groups(outcomes, lambda o: group_of[o.query_id], TRUNCATION_GROUPS)
    truncated = [o for o in outcomes if group_of[o.query_id] != NOT_TRUNCATED]
    rows = [[name, *metric_cells(summary)] for name, summary in groups.items()]
    rows.append(["truncated (both)", *metric_cells(summarize(truncated) if truncated else None)])
    return table(["Relevant chunk at 1200/200", "Queries", "R@1", "R@3", "R@5", "MRR"], rows)


def truncation_section(production: ConfigRun) -> str:
    group_of = truncation_groups(production)
    outcomes = [r.outcome for r in production.results]
    # Cross-language queries fail for another reason and sit almost only in the
    # "not truncated" group; comparing within same-language queries removes that.
    same_language = [o for o in outcomes if o.direction in SAME_LANGUAGE_DIRECTIONS]
    return ("All queries:\n\n" + _truncation_table(outcomes, group_of)
            + "\n\nSame-language queries only (" + ", ".join(SAME_LANGUAGE_DIRECTIONS)
            + "):\n\n" + _truncation_table(same_language, group_of))


def experiment_section(runs: Sequence[ConfigRun]) -> str:
    rows = []
    for run in runs:
        stats = analysis.chunk_stats(run.kb)
        summary = summarize([r.outcome for r in run.results])
        by_language = "/".join(str(stats.truncated_by_language[lang]) for lang in LANGUAGES)
        rows.append([stats.chunk_size, stats.chunk_overlap, stats.chunks,
                     f"{stats.average_chars:.0f}", stats.max_tokens,
                     f"{stats.truncated} ({by_language})", *metric_cells(summary)[1:]])
    table_text = table(["Chunk size", "Overlap", "Chunks", "Avg chars", "Max tokens",
                        "Truncated (en/zh/mixed)", "R@1", "R@3", "R@5", "MRR"], rows)
    # How the queries with truncated evidence at 1200/200 fare with other chunk sizes.
    group_at_production = truncation_groups(runs[0])
    watched = [query_id for query_id, group in group_at_production.items()
               if group != NOT_TRUNCATED]
    rank_rows = []
    for query_id in watched:
        ranks = []
        for run in runs:
            outcome = next(r.outcome for r in run.results if r.query.query_id == query_id)
            ranks.append(min((x for x in outcome.ranks if x is not None), default="-"))
        rank_rows.append([query_id, group_at_production[query_id], *ranks])
    headers = ["Query", "At 1200/200", *(f"rank @{r.kb.config.chunk_size}" for r in runs)]
    return table_text + "\n\nQueries with truncated evidence at 1200/200:\n\n" + table(
        headers, rank_rows)


def cross_language_section(production: ConfigRun) -> str:
    rows = []
    for direction in ("en->en", "zh->zh", "zh->en", "en->zh"):
        diagnoses = [analysis.diagnose(production.kb, r) for r in production.results
                     if r.outcome.direction == direction]
        ranks = [d.first_relevant_rank for d in diagnoses]
        in_query_language = sum(d.query_language_in_top for d in diagnoses)
        rows.append([direction, len(diagnoses), " ".join(str(r or "-") for r in ranks),
                     f"{in_query_language}/{len(diagnoses) * analysis.TOP_N_FOR_LANGUAGE_MIX}"])
    return table(["Direction", "Queries", "First relevant rank per query",
                  f"Top-{analysis.TOP_N_FOR_LANGUAGE_MIX} results in the query's language"], rows)


def performance_section(runs: Sequence[ConfigRun], total_seconds: float) -> str:
    rows = []
    for run in runs:
        searches = [r.search_seconds * 1000 for r in run.results]
        rows.append([f"{run.kb.config.chunk_size}/{run.kb.config.chunk_overlap}",
                     len(run.kb.chunks), len(run.results), f"{run.kb.embed_seconds:.2f}",
                     f"{sum(searches) / len(searches):.1f}", f"{max(searches):.1f}",
                     f"{run.seconds:.2f}"])
    return (table(["Config", "Chunks", "Queries", "Embedding s", "Search ms (mean)",
                   "Search ms (max)", "Build + evaluate s"], rows)
            + f"\n\nTotal report time: {total_seconds:.1f} s (includes model loading and "
            "token analysis). Search time includes embedding the question.")


def main() -> int:
    started = time.perf_counter()
    logging.getLogger("app").setLevel(logging.WARNING)
    spec = get_model_spec(EmbeddingConfig(cache_dir=MODEL_CACHE_DIR).model_name)
    if not model_is_cached(spec, MODEL_CACHE_DIR):
        print("The embedding model is not cached yet. Embed any document once (it downloads "
              "the pinned model), then run the report again.", file=sys.stderr)
        return 1
    validate_dataset()
    provider = FastEmbedProvider(EmbeddingConfig(cache_dir=MODEL_CACHE_DIR))

    runs = [run_config(provider, config) for config in EXPERIMENT_CONFIGS]
    production = runs[0]
    repeat = run_config(provider, PRODUCTION)  # fresh folder, fresh embeddings
    identical = analysis.fingerprint(repeat.results) == analysis.fingerprint(production.results)
    token_text = token_section(provider, production)

    sections = [
        ("Dataset", dataset_section(production)),
        ("Metrics at 1200/200 (production default)", metrics_section(production)),
        ("Cross-language diagnosis", cross_language_section(production)),
        ("Per query at 1200/200", query_section(production)),
        ("Tokens and truncation", token_text),
        ("Retrieval by truncation of the relevant chunk", truncation_section(production)),
        ("Chunk size experiment", experiment_section(runs)),
        ("Determinism", "Second independent build + evaluation at 1200/200: rankings, "
                        "full-precision scores and labels identical: "
                        f"**{'yes' if identical else 'NO'}**"),
    ]
    sections.append(("Performance", performance_section([*runs, repeat],
                                                        time.perf_counter() - started)))
    print(f"# DocuBot retrieval evaluation\n\nModel {spec.name}@{spec.revision[:12]}. "
          "A development evaluation set, not a general benchmark.")
    for title, body in sections:
        print(f"\n## {title}\n\n{body}")
    return 0 if identical else 2


if __name__ == "__main__":
    sys.exit(main())
