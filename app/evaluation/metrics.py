"""Retrieval metrics: pure functions, no model, no database.

Vocabulary:
    item  - one golden relevant label of a query (see golden.py). A query
            with two relevant labels has two items.
    rank  - position in the search results, 1 = best. The rank of an item
            is the best rank of any chunk that contains its evidence, or
            None when no such chunk was returned.

Recall@K = (items with rank <= K) / (all items of the query)
    With one item this is 1 or 0: "is the evidence in the top K?". With two
    items and only one found it is 0.5. The denominator is the number of
    ITEMS, never the number of chunks: when overlap copies the evidence into
    two chunks, finding either chunk finds the item once.

Reciprocal rank = 1 / (rank of the best-ranked item), 0 if none was returned.
    rank 1 -> 1.0, rank 2 -> 0.5, rank 4 -> 0.25, not returned -> 0.
MRR (Mean Reciprocal Rank) = the average reciprocal rank over the queries.

Every summary is an average over queries; each query counts once.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

RECALL_KS = (1, 3, 5)

Rank = int | None  # None = not returned


def item_ranks(ranked_chunk_ids: Sequence[str], item_chunk_sets: Sequence[set[str]]) -> tuple[Rank, ...]:
    """The rank of every item: the best position of any of its chunks, or None.

    ranked_chunk_ids is the search result order (best first, no duplicates).
    item_chunk_sets[i] holds the chunk IDs that contain item i's evidence.
    """
    if len(set(ranked_chunk_ids)) != len(ranked_chunk_ids):
        raise ValueError("A chunk appears twice in the ranking.")
    position = {chunk_id: rank for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)}
    ranks: list[Rank] = []
    for chunk_ids in item_chunk_sets:
        found = [position[chunk_id] for chunk_id in chunk_ids if chunk_id in position]
        ranks.append(min(found) if found else None)
    return tuple(ranks)


def recall_at_k(ranks: Sequence[Rank], k: int) -> float:
    """Share of the query's items found in the top k."""
    if not ranks:
        raise ValueError("A query needs at least one relevant item.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def reciprocal_rank(ranks: Sequence[Rank]) -> float:
    """1 / rank of the best-ranked item, or 0.0 if no item was returned."""
    if not ranks:
        raise ValueError("A query needs at least one relevant item.")
    found = [rank for rank in ranks if rank is not None]
    if any(rank < 1 for rank in found):
        raise ValueError("Ranks start at 1.")
    return 1.0 / min(found) if found else 0.0


@dataclass(frozen=True)
class QueryOutcome:
    """What the metrics need to know about one evaluated query."""

    query_id: str
    direction: str  # e.g. "zh->en"
    ranks: tuple[Rank, ...]  # one per relevant item


@dataclass(frozen=True)
class MetricSummary:
    queries: int
    recall: dict[int, float]  # k -> mean Recall@k
    mrr: float


def summarize(outcomes: Sequence[QueryOutcome]) -> MetricSummary:
    """Mean Recall@1/3/5 and MRR over the queries (each query counts once)."""
    if not outcomes:
        raise ValueError("At least one query is needed.")
    count = len(outcomes)
    recall = {k: sum(recall_at_k(o.ranks, k) for o in outcomes) / count for k in RECALL_KS}
    mrr = sum(reciprocal_rank(o.ranks) for o in outcomes) / count
    return MetricSummary(queries=count, recall=recall, mrr=mrr)


def summarize_groups(
    outcomes: Iterable[QueryOutcome],
    group_of: Callable[[QueryOutcome], str],
    order: Sequence[str],
) -> dict[str, MetricSummary | None]:
    """One summary per group, in `order`. A group without queries maps to None.

    Every group in `order` appears in the result, so a category can never
    silently disappear from a report. A query whose group is not in `order`
    is an error, not something to drop.
    """
    groups: dict[str, list[QueryOutcome]] = {name: [] for name in order}
    for outcome in outcomes:
        name = group_of(outcome)
        if name not in groups:
            raise ValueError(f"Unknown group {name!r} for query {outcome.query_id!r}.")
        groups[name].append(outcome)
    return {name: summarize(items) if items else None for name, items in groups.items()}
