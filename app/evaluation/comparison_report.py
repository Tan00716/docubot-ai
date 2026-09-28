"""Markdown report of the Batch 6B model comparison (pure formatting, no model).

Every number comes from the ModelResults. Deltas are always "candidate
minus the E5-small baseline" on the same questions. The per-query tables
show the cross-language questions for every model; they are evaluation
output only and are never written to application logs.
"""

from collections.abc import Sequence

from app.evaluation.candidates import get_candidate
from app.evaluation.comparison import (
    CROSS_LANGUAGE_DIRECTIONS,
    METRIC_NAMES,
    ModelResult,
    by_direction,
    cross_language,
    cross_language_profile,
    deltas,
    metric_values,
    overall,
    truncation,
)
from app.evaluation.golden import DIRECTIONS
from app.evaluation.metrics import MetricSummary, QueryOutcome, summarize
from app.evaluation.dataset_versions import verify_dataset

MEGABYTE, GIGABYTE = 1024**2, 1024**3


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def signed(value: float) -> str:
    return f"{value:+.3f}"


def size(value: object) -> str:
    if not isinstance(value, int):
        return "not measured"
    return f"{value / GIGABYTE:.2f} GB" if value >= GIGABYTE else f"{value / MEGABYTE:.0f} MB"


def prefix_text(prefix: str) -> str:
    return f"`{prefix}`" if prefix else "none"


def metric_row(summary: MetricSummary, baseline: MetricSummary | None) -> list[str]:
    values = [f"{v:.3f}" for v in metric_values(summary).values()]
    if baseline is None:
        return values + ["-"] * len(METRIC_NAMES)
    return values + [signed(v) for v in deltas(summary, baseline).values()]


def success_counts(result: ModelResult, predicate=lambda query: True) -> str:
    selected = [query for query in result.queries if predicate(query)]
    if not selected:
        return "0/0, 0/0, 0/0"
    return ", ".join(
        f"{sum(any(rank is not None and rank <= k for rank in query.ranks) for query in selected)}"
        f"/{len(selected)}" for k in (1, 3, 5))


def models_section(results: Sequence[ModelResult], skipped: dict[str, str]) -> str:
    rows = []
    for result in results:
        candidate = get_candidate(result.key)
        rows.append([result.model_name, result.dimension, result.max_tokens, candidate.languages,
                     f"query {prefix_text(result.query_prefix)}, passage "
                     f"{prefix_text(result.passage_prefix)}", result.pooling, candidate.runtime,
                     f"`{result.revision[:12]}`"])
    text = table(["Model", "Dimension", "Max tokens", "Languages", "Prefix contract", "Pooling",
                  "Runtime", "Revision"], rows)
    if skipped:
        text += "\n\nNot evaluated:\n\n" + "\n".join(
            f"- **{get_candidate(key).spec.name}**: {reason}" for key, reason in skipped.items())
    return text


def overall_section(results: Sequence[ModelResult]) -> str:
    baseline = overall(results[0])
    rows = [[r.model_name, *metric_row(overall(r), None if r is results[0] else baseline),
             success_counts(r)]
            for r in results]
    return table(["Model", *METRIC_NAMES, *(f"Δ {m}" for m in METRIC_NAMES),
                  "Query hits @1/@3/@5"], rows)


def cross_section(results: Sequence[ModelResult]) -> str:
    baseline = cross_language(results[0])
    rows = [[r.model_name, cross_language(r).queries,
             *metric_row(cross_language(r), None if r is results[0] else baseline),
             success_counts(r, lambda q: _direction_is_cross(q.direction, r))]
            for r in results]
    grouping = (" + ".join(CROSS_LANGUAGE_DIRECTIONS) if results[0].dataset_version == "retrieval_eval_v1"
                else "all directions where query and target languages differ")
    return ("Cross-language = " + grouping
            + " (each question counts once).\n\n"
            + table(["Model", "Queries", *(f"Cross {m}" for m in METRIC_NAMES),
                     *(f"Δ {m}" for m in METRIC_NAMES), "Query hits @1/@3/@5"], rows))


def direction_section(results: Sequence[ModelResult]) -> str:
    baseline = by_direction(results[0])
    rows = []
    for direction in DIRECTIONS:
        for result in results:
            summary = by_direction(result)[direction]
            base = None if result is results[0] else baseline[direction]
            values = metric_row(summary, base)
            hits = success_counts(result, lambda q: q.direction == direction)
            rows.append([direction, result.model_name, summary.queries, *values[:4], values[7], hits])
    return table(["Direction", "Model", "Queries", *METRIC_NAMES, "Δ MRR",
                  "Query hits @1/@3/@5"], rows)


def profile_section(results: Sequence[ModelResult]) -> str:
    rows = []
    for result in results:
        p = cross_language_profile(result)
        rows.append([result.model_name, p.queries, p.at_rank_1, p.in_top_3, p.in_top_5,
                     " ".join(str(rank or "-") for rank in p.first_ranks),
                     f"{p.query_language_in_top}/{p.top_slots}", p.truncated_evidence])
    cross_ids = [query.query_id for query in results[0].queries
                 if _direction_is_cross(query.direction, results[0])]
    return ("First relevant rank per cross-language question, in dataset order "
            f"({', '.join(cross_ids)}).\n\n"
            + table(["Model", "Queries", "Rank 1", "Top 3", "Top 5", "First relevant ranks",
                     "Top-5 results in the question's language", "Relevant chunk truncated"],
                    rows))


def _direction_is_cross(direction: str, result: ModelResult) -> bool:
    if result.dataset_version == "retrieval_eval_v1":
        return direction in CROSS_LANGUAGE_DIRECTIONS
    return direction.split("->")[0] != direction.split("->")[1]


def truncation_section(results: Sequence[ModelResult]) -> str:
    rows = []
    query_rows = []
    for result in results:
        t = truncation(result)
        cells = [f"{t[k].truncated}/{t[k].chunks} ({t[k].percent:.0f}%)"
                 for k in ("en", "zh", "mixed", "total")]
        longest = max(c.tokens for c in result.chunks)
        rows.append([result.model_name, result.max_tokens, *cells, longest])
        for group in ("not truncated", "truncated, evidence inside window",
                      "truncated, evidence beyond window"):
            group_outcomes = [QueryOutcome(q.query_id, q.direction, q.ranks)
                              for q in result.queries if q.truncation_group == group]
            if group_outcomes:
                summary = summarize(group_outcomes)
                query_rows.append([result.model_name, group, summary.queries,
                                   *(f"{summary.recall[k]:.3f}" for k in (1, 3, 5)),
                                   f"{summary.mrr:.3f}"])
    query_table = table(["Model", "Relevant evidence group", "Queries", "R@1", "R@3", "R@5", "MRR"], query_rows)
    return (f"The same {len(results[0].chunks)} chunks (1200/200) for every model; a chunk is truncated when its "
            "model input (prefix included) has more tokens than the model reads.\n\n"
            + table(["Model", "Token limit", "English", "Chinese", "Mixed", "Total",
                     "Longest chunk (tokens)"], rows) + "\n\nQuery outcomes grouped by whether relevant evidence is model-readable:\n\n"
            + query_table)


def resources_section(results: Sequence[ModelResult], env: dict[str, str]) -> str:
    rows = []
    for result in results:
        r = result.resources
        rows.append([result.model_name, size(get_candidate(result.key).download_bytes),
                     size(r["cache_bytes"]), f"{r['load_seconds']:.1f} s",
                     f"{r['embed_seconds']:.1f} s",
                     f"{r['search_ms_mean']:.0f} ms (max {r['search_ms_max']:.0f})",
                     size(r["peak_working_set"]), size(r["peak_private_bytes"])])
    batch = results[0].resources["embedding_batch_size"]
    setup = "; ".join(f"{key} {value}" for key, value in env.items())
    return (table(["Model", "Cold download", "Cache on disk", "Model load",
                   f"Embed {len(results[0].chunks)} chunks",
                   "Query embed + search (mean)", "Peak working set", "Peak private memory"],
                  rows)
            + f"\n\nSetup (identical for every model): {setup}; embedding batch size {batch} "
            "(the application default; every passage is its own document). Each model ran in "
            "its own fresh process. Timings are single-run measurements on this laptop; treat "
            "them as rough indications, not benchmarks.")


def checks_section(results: Sequence[ModelResult]) -> str:
    names = list(results[0].checks)
    rows = [[name, *("yes" if r.checks.get(name) else "**NO**" for r in results)]
            for name in names]
    return table(["Contract check", *(r.model_name for r in results)], rows)


def diagnostics_section(results: Sequence[ModelResult]) -> str:
    dataset = verify_dataset(results[0].dataset_version)
    text_of = {q.query_id: q for q in dataset.queries}
    language_of = {p.key: p.language for p in dataset.passages}
    parts = []
    for result in results:
        passage_of = {c.chunk_id: c.passage_key for c in result.chunks}
        rows = []
        for query in result.queries:
            if not _direction_is_cross(query.direction, result):
                continue
            golden = text_of[query.query_id]
            evidence = "; ".join(f"{e.passage_key}: \"{e.text}\"" for e in golden.relevant)
            top = "<br>".join(
                f"{'**' if hit.relevant else ''}{hit.language} {passage_of[hit.chunk_id]} "
                f"`{hit.chunk_id[:8]}…{hit.chunk_id[-5:]}` {hit.score:.4f}"
                f"{' [HARD NEGATIVE]' if hit.hard_negative else ''}"
                f"{'**' if hit.relevant else ''}" for hit in query.top)
            target = language_of[golden.relevant[0].passage_key]
            rows.append([query.query_id, golden.text, f"{query.query_language}→{target}",
                         evidence, query.first_relevant_rank or "-", top])
        parts.append(f"**{result.model_name}** (relevant chunks in bold)\n\n" + table(
            ["Query", "Question", "Direction", "Expected evidence", "First relevant rank",
             "Top 5: language, passage, chunk ID, score"], rows))
    return "\n\n".join(parts)


def render(results: Sequence[ModelResult], skipped: dict[str, str], env: dict[str, str],
           seconds: float) -> str:
    dataset = verify_dataset(results[0].dataset_version)
    sections = [
        ("Models", models_section(results, skipped)),
        ("Overall", overall_section(results)),
        ("Cross-language (primary metric)", cross_section(results)),
        ("Language breakdown", direction_section(results)),
        ("Where the cross-language change comes from", profile_section(results)),
        ("Truncation", truncation_section(results)),
        ("Resources", resources_section(results, env)),
        ("Vector and input contract checks (real runs)", checks_section(results)),
        ("Per-query cross-language diagnostics", diagnostics_section(results)),
    ]
    batch = "Batch 6B" if dataset.version == "retrieval_eval_v1" else "Batch 6C"
    head = (f"# DocuBot embedding model comparison ({batch})\n\n"
            f"A development evaluation set ({len(results[0].chunks)} chunks, {len(dataset.queries)} questions; "
            f"{dataset.version}, SHA-256 {dataset.sha256}), not a general benchmark. "
            "Only the embedding model changes; chunking 1200/200, exact search, ranking, "
            "top_k and labels are the Batch 6A ones. Deltas are candidate minus "
            f"{results[0].model_name}. One question changes a direction's Recall by 0.2-1.0. "
            f"Total time {seconds / 60:.1f} min.")
    return head + "".join(f"\n\n## {title}\n\n{body}" for title, body in sections)
