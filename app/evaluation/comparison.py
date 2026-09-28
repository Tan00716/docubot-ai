"""Controlled embedding model comparison: records and metrics (pure, no model).

Only the embedding model changes between runs. Everything else is fixed:

    passages + questions + labels   the Batch 6A dataset, frozen by DATASET_SHA256
    chunking                        production ChunkingConfig() = 1200 / 200
    search + ranking + top_k        app.search.service.search, top_k = MAX_TOP_K
    metrics                         app.evaluation.metrics (Recall@1/3/5, MRR)

A ModelResult is what one model run produced. It is plain data, so it can be
written to JSON by the process that ran the model and read back by the
report. validate_result() refuses a result with a missing query, a missing
language direction, another dataset or another chunk count, so a failed
question can never quietly disappear from a comparison.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from app.embeddings.config import EmbeddingModelSpec
from app.embeddings.vectors import text_sha256
from app.evaluation.corpus import LANGUAGES, PASSAGES, Passage
from app.evaluation.golden import DIRECTIONS, QUERIES, GoldenQuery, direction_of
from app.evaluation.metrics import MetricSummary, QueryOutcome, Rank, summarize, summarize_groups
from app.evaluation.runner import NOT_TRUNCATED, KnowledgeBase, QueryResult, relevant_chunk_ids

# SHA-256 of the Batch 6A dataset (dataset_fingerprint()). Changing a passage,
# a question, a label or a direction changes it, and the comparison refuses to run.
DATASET_SHA256 = "7b8a9e13b131e4042cc13f4d77aeb66a876a2f947900f066a97b97cb8ee921c7"
EXPECTED_CHUNKS = 42  # at the production chunking 1200 / 200
EXPECTED_QUERIES = 33
EXPECTED_DIRECTION_COUNTS = {
    "en->en": 5, "zh->zh": 9, "zh->en": 5, "en->zh": 5, "mixed->en": 3, "mixed->zh": 3,
    "zh->mixed": 1, "en->mixed": 1, "mixed->mixed": 1,
}
# The primary metric of Batch 6B: question and answer in different languages.
CROSS_LANGUAGE_DIRECTIONS = ("zh->en", "en->zh")
TOP_N = 5  # how many results the per-query diagnostics keep
METRIC_NAMES = ("R@1", "R@3", "R@5", "MRR")


class IncompleteResultError(ValueError):
    """A model result does not cover exactly the frozen dataset."""


def dataset_fingerprint(passages: Sequence[Passage] = PASSAGES,
                        queries: Sequence[GoldenQuery] = QUERIES) -> str:
    """SHA-256 over every passage, question, label and direction, in order."""
    data = {
        "passages": [[p.key, p.language, p.text] for p in passages],
        "queries": [[q.query_id, q.language, q.text,
                     [[e.passage_key, e.text] for e in q.relevant],
                     [[e.passage_key, e.text] for e in q.secondary],
                     direction_of(q, passages)] for q in queries],
        "directions": list(DIRECTIONS),
    }
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- Records ------------------------------------------------------------------


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    score: float  # full precision
    language: str
    relevant: bool


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    direction: str
    query_language: str
    ranks: tuple[Rank, ...]  # one per relevant item
    top: tuple[Hit, ...]  # the first TOP_N results
    truncation_group: str
    search_seconds: float

    @property
    def first_relevant_rank(self) -> int | None:
        found = [rank for rank in self.ranks if rank is not None]
        return min(found) if found else None


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    passage_key: str
    language: str
    text_sha256: str
    tokens: int  # model input tokens before the model's cut
    truncated: bool


@dataclass(frozen=True)
class ModelResult:
    key: str
    model_name: str
    revision: str
    dimension: int
    max_tokens: int
    pooling: str
    passage_prefix: str
    query_prefix: str
    dataset_sha256: str
    chunks: tuple[ChunkRecord, ...]
    queries: tuple[QueryRecord, ...]
    checks: dict[str, bool]  # vector / input contract checks of the real run
    resources: dict[str, float | int | str | None]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "ModelResult":
        data = json.loads(text)
        chunks = tuple(ChunkRecord(**chunk) for chunk in data.pop("chunks"))
        queries = tuple(
            QueryRecord(**(query | {"ranks": tuple(query["ranks"]),
                                    "top": tuple(Hit(**hit) for hit in query["top"])}))
            for query in data.pop("queries"))
        return cls(**data, chunks=chunks, queries=queries)


def collect_result(key: str, spec: EmbeddingModelSpec, kb: KnowledgeBase,
                   results: Sequence[QueryResult], *, checks: dict[str, bool],
                   resources: dict[str, float | int | str | None]) -> ModelResult:
    """Turn one runner.evaluate() run into plain data."""
    queries = []
    for result in results:
        relevant = relevant_chunk_ids(kb, result.query)
        top = tuple(Hit(chunk_id, score, kb.chunks[chunk_id].language, chunk_id in relevant)
                    for chunk_id, score in result.ranking[:TOP_N])
        queries.append(QueryRecord(
            query_id=result.query.query_id, direction=result.outcome.direction,
            query_language=result.query.language, ranks=result.outcome.ranks, top=top,
            truncation_group=result.truncation_group, search_seconds=result.search_seconds,
        ))
    chunks = tuple(ChunkRecord(c.chunk_id, c.passage_key, c.language, text_sha256(c.text),
                               c.tokens, c.truncated) for c in kb.chunks.values())
    return ModelResult(
        key=key, model_name=spec.name, revision=spec.revision, dimension=spec.dimension,
        max_tokens=spec.max_tokens, pooling=spec.pooling, passage_prefix=spec.passage_prefix,
        query_prefix=spec.query_prefix, dataset_sha256=dataset_fingerprint(),
        chunks=chunks, queries=tuple(queries), checks=dict(checks), resources=dict(resources),
    )


# --- Integrity ----------------------------------------------------------------


def validate_result(result: ModelResult) -> None:
    """Raise IncompleteResultError unless the result covers exactly the frozen dataset."""
    if result.dataset_sha256 != DATASET_SHA256:
        raise IncompleteResultError(f"{result.key}: evaluated on another dataset.")
    if [q.query_id for q in result.queries] != [q.query_id for q in QUERIES]:
        raise IncompleteResultError(f"{result.key}: queries missing, added or reordered.")
    counts = {name: 0 for name in EXPECTED_DIRECTION_COUNTS}
    for query in result.queries:
        if query.direction not in counts:
            raise IncompleteResultError(f"{result.key}: unknown direction {query.direction!r}.")
        counts[query.direction] += 1
    if counts != EXPECTED_DIRECTION_COUNTS:
        raise IncompleteResultError(f"{result.key}: language directions changed: {counts}.")
    if len(result.chunks) != EXPECTED_CHUNKS:
        raise IncompleteResultError(f"{result.key}: {len(result.chunks)} chunks, "
                                    f"expected {EXPECTED_CHUNKS}.")


def check_same_chunks(results: Sequence[ModelResult]) -> None:
    """Every model must have searched exactly the same chunks (a controlled experiment)."""
    def identity(result: ModelResult) -> list[tuple[str, str, str, str]]:
        return [(c.chunk_id, c.passage_key, c.language, c.text_sha256) for c in result.chunks]

    for result in results[1:]:
        if identity(result) != identity(results[0]):
            raise IncompleteResultError(f"{result.key} searched other chunks than "
                                        f"{results[0].key}.")


# --- Metrics ------------------------------------------------------------------


def outcomes(result: ModelResult,
             directions: Sequence[str] | None = None) -> list[QueryOutcome]:
    return [QueryOutcome(q.query_id, q.direction, q.ranks) for q in result.queries
            if directions is None or q.direction in directions]


def overall(result: ModelResult) -> MetricSummary:
    return summarize(outcomes(result))


def by_direction(result: ModelResult) -> dict[str, MetricSummary | None]:
    """One summary per direction, every direction present (see summarize_groups)."""
    return summarize_groups(outcomes(result), lambda o: o.direction, DIRECTIONS)


def cross_language(result: ModelResult) -> MetricSummary:
    """zh->en and en->zh together; each question counts once."""
    selected = outcomes(result, CROSS_LANGUAGE_DIRECTIONS)
    present = {o.direction for o in selected}
    if present != set(CROSS_LANGUAGE_DIRECTIONS):
        raise IncompleteResultError(f"{result.key}: a cross-language direction is missing.")
    return summarize(selected)


def metric_values(summary: MetricSummary) -> dict[str, float]:
    return {"R@1": summary.recall[1], "R@3": summary.recall[3], "R@5": summary.recall[5],
            "MRR": summary.mrr}


def deltas(candidate: MetricSummary, baseline: MetricSummary) -> dict[str, float]:
    """candidate - baseline for every metric (positive = the candidate is better)."""
    if candidate.queries != baseline.queries:
        raise ValueError("Summaries over different numbers of queries cannot be compared.")
    base = metric_values(baseline)
    return {name: value - base[name] for name, value in metric_values(candidate).items()}


@dataclass(frozen=True)
class Truncation:
    truncated: int
    chunks: int

    @property
    def percent(self) -> float:
        return 100.0 * self.truncated / self.chunks if self.chunks else 0.0


def truncation(result: ModelResult) -> dict[str, Truncation]:
    """Truncated chunks per chunk language, plus "total"."""
    table = {language: Truncation(sum(c.truncated for c in result.chunks if c.language == language),
                                  sum(c.language == language for c in result.chunks))
             for language in LANGUAGES}
    table["total"] = Truncation(sum(c.truncated for c in result.chunks), len(result.chunks))
    return table


@dataclass(frozen=True)
class CrossLanguageProfile:
    """Where a cross-language change comes from (see the Batch 6B report)."""

    queries: int
    first_ranks: tuple[int | None, ...]  # per question, dataset order
    at_rank_1: int
    in_top_3: int
    in_top_5: int
    query_language_in_top: int  # top-N results written in the QUESTION's language
    top_slots: int  # queries * TOP_N
    truncated_evidence: int  # questions whose relevant chunk was truncated


def cross_language_profile(result: ModelResult) -> CrossLanguageProfile:
    selected = [q for q in result.queries if q.direction in CROSS_LANGUAGE_DIRECTIONS]
    ranks = tuple(q.first_relevant_rank for q in selected)

    def within(k: int) -> int:
        return sum(rank is not None and rank <= k for rank in ranks)

    return CrossLanguageProfile(
        queries=len(selected), first_ranks=ranks, at_rank_1=within(1), in_top_3=within(3),
        in_top_5=within(5),
        query_language_in_top=sum(hit.language == q.query_language
                                  for q in selected for hit in q.top),
        top_slots=sum(len(q.top) for q in selected),
        truncated_evidence=sum(q.truncation_group != NOT_TRUNCATED for q in selected),
    )
