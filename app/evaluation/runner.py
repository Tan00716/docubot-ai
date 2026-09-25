"""Run the golden questions through the REAL DocuBot retrieval pipeline.

For one ChunkingConfig:

    every passage -> one .txt document in a temporary folder
    -> process_document -> chunk_document (the production chunker)
    -> embed_document (the given provider) -> SQLite
    -> for every golden question: app.search.service.search(top_k=MAX_TOP_K)
    -> map the returned chunks to golden items -> QueryResult

Nothing here re-implements retrieval: scores and ranks come from the same
search() function that serves POST /search. The caller owns the temporary
folder; nothing is written anywhere else, and nothing is downloaded (the
provider decides that; the report only runs with a cached model).

Document IDs are derived from the passage key (not random), so chunk IDs,
the chunk_id tie-breaker and therefore the whole ranking are reproducible.
"""

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.document_processing import database, processor
from app.document_processing.chunking import ChunkingConfig
from app.embeddings.provider import EmbeddingProvider
from app.embeddings.service import embed_document
from app.evaluation.corpus import PASSAGES, Language, Passage
from app.evaluation.golden import QUERIES, Evidence, GoldenQuery, direction_of
from app.evaluation.metrics import QueryOutcome, Rank, item_ranks
from app.search import storage as search_storage
from app.search.models import MAX_TOP_K
from app.search.service import search

# How a query's evidence was seen by the model (see truncation_group()).
NOT_TRUNCATED = "not truncated"
TRUNCATED_EVIDENCE_READ = "truncated, evidence inside window"
TRUNCATED_EVIDENCE_CUT = "truncated, evidence beyond window"
TRUNCATION_GROUPS = (NOT_TRUNCATED, TRUNCATED_EVIDENCE_READ, TRUNCATED_EVIDENCE_CUT)


class TokenCountingProvider(EmbeddingProvider, Protocol):
    """An embedding provider that can also report exact token counts."""

    def count_tokens(self, model_input: str) -> int: ...


class DatasetError(ValueError):
    """The golden labels do not fit the chunks (e.g. evidence split across chunks)."""


@dataclass(frozen=True)
class EvalChunk:
    chunk_id: str
    passage_key: str
    language: Language
    text: str
    tokens: int  # tokens of "passage: " + text, before the model's cut
    truncated: bool  # as stored by the embedding pipeline


@dataclass(frozen=True)
class KnowledgeBase:
    config: ChunkingConfig
    db_path: Path
    passages: tuple[Passage, ...]  # what was uploaded (decides each query's direction)
    chunks: dict[str, EvalChunk]  # chunk_id -> chunk, sorted by chunk_id
    embed_seconds: float


@dataclass(frozen=True)
class QueryResult:
    query: GoldenQuery
    ranking: tuple[tuple[str, float], ...]  # (chunk_id, full-precision score), best first
    outcome: QueryOutcome  # ranks of the relevant items
    secondary_ranks: tuple[Rank, ...]
    truncation_group: str
    search_seconds: float


def passage_document_id(passage_key: str) -> str:
    """A stable 32-hex-character document ID for a passage."""
    return hashlib.sha256(passage_key.encode("utf-8")).hexdigest()[:32]


def _add_passage(passage: Passage, upload_dir: Path, processed_dir: Path, db_path: Path,
                 config: ChunkingConfig) -> str:
    document_id = passage_document_id(passage.key)
    data = passage.text.encode("utf-8")
    stored_filename = f"{document_id}.txt"
    (upload_dir / stored_filename).write_bytes(data)
    database.create_document(
        db_path, document_id=document_id, original_filename=f"{passage.key}.txt",
        stored_filename=stored_filename, extension=".txt", content_type="text/plain",
        size_bytes=len(data),
    )
    processor.process_document(document_id, upload_dir=upload_dir,
                               processed_dir=processed_dir, db_path=db_path)
    processor.chunk_document(document_id, upload_dir=upload_dir, processed_dir=processed_dir,
                             db_path=db_path, config=config)
    return document_id


def build_knowledge_base(root: Path, provider: TokenCountingProvider, config: ChunkingConfig,
                         passages: Sequence[Passage] = PASSAGES) -> KnowledgeBase:
    """Upload, process, chunk and embed every passage under `root` (a fresh folder)."""
    upload_dir, processed_dir = root / "uploads", root / "processed"
    db_path = root / "metadata" / "evaluation.db"
    upload_dir.mkdir(parents=True)
    passage_of: dict[str, Passage] = {}
    for passage in passages:
        passage_of[_add_passage(passage, upload_dir, processed_dir, db_path, config)] = passage

    started = time.perf_counter()
    for document_id in passage_of:
        embed_document(document_id, upload_dir=upload_dir, db_path=db_path, provider=provider)
    embed_seconds = time.perf_counter() - started

    # The truncated flag exactly as the pipeline stored it (read-only query).
    candidates, stale_count = search_storage.load_candidates(db_path, provider.contract)
    if stale_count:
        raise DatasetError("Freshly embedded chunks must not be stale.")
    chunks: dict[str, EvalChunk] = {}
    for candidate in sorted(candidates, key=lambda c: c.chunk.chunk_id):
        chunk = candidate.chunk
        passage = passage_of[chunk.document_id]
        chunks[chunk.chunk_id] = EvalChunk(
            chunk_id=chunk.chunk_id, passage_key=passage.key, language=passage.language,
            text=chunk.text,
            tokens=provider.count_tokens(provider.contract.passage_input(chunk.text)),
            truncated=candidate.truncated,
        )
    return KnowledgeBase(config=config, db_path=db_path, passages=tuple(passages),
                         chunks=chunks, embed_seconds=embed_seconds)


def evidence_chunks(kb: KnowledgeBase, evidence: Evidence) -> list[EvalChunk]:
    """Every chunk of the evidence's passage that contains the whole evidence text."""
    found = [chunk for chunk in kb.chunks.values()
             if chunk.passage_key == evidence.passage_key and evidence.text in chunk.text]
    if not found:
        raise DatasetError(
            f"Evidence of {evidence.passage_key!r} is in no chunk at chunk_size "
            f"{kb.config.chunk_size} (split across a chunk boundary?)."
        )
    return found


def evidence_is_read(provider: TokenCountingProvider, chunk: EvalChunk, evidence: Evidence) -> bool:
    """Whether the model read the evidence: its end lies inside the token window."""
    if not chunk.truncated:
        return True
    end = chunk.text.index(evidence.text) + len(evidence.text)
    prefix_input = provider.contract.passage_input(chunk.text[:end])
    return provider.count_tokens(prefix_input) <= provider.contract.max_tokens


def truncation_group(kb: KnowledgeBase, provider: TokenCountingProvider,
                     query: GoldenQuery) -> str:
    """How the model saw the query's relevant evidence (the worst item decides).

    For each item, its best chunk counts: an untruncated chunk beats a
    truncated one that still shows the evidence, which beats a chunk where
    the evidence lies after the model's token window.
    """
    order = {name: index for index, name in enumerate(TRUNCATION_GROUPS)}
    worst = NOT_TRUNCATED
    for evidence in query.relevant:
        statuses = []
        for chunk in evidence_chunks(kb, evidence):
            if not chunk.truncated:
                statuses.append(NOT_TRUNCATED)
            elif evidence_is_read(provider, chunk, evidence):
                statuses.append(TRUNCATED_EVIDENCE_READ)
            else:
                statuses.append(TRUNCATED_EVIDENCE_CUT)
        best = min(statuses, key=order.__getitem__)
        worst = max(worst, best, key=order.__getitem__)
    return worst


def _ranks_of(kb: KnowledgeBase, ranked_ids: list[str],
              evidence: Sequence[Evidence]) -> tuple[Rank, ...]:
    sets = [{chunk.chunk_id for chunk in evidence_chunks(kb, item)} for item in evidence]
    return item_ranks(ranked_ids, sets)


def evaluate_query(kb: KnowledgeBase, provider: TokenCountingProvider,
                   query: GoldenQuery) -> QueryResult:
    started = time.perf_counter()
    found = search(query.text, db_path=kb.db_path, provider=provider, top_k=MAX_TOP_K)
    search_seconds = time.perf_counter() - started
    ranking = tuple((result.chunk.chunk_id, result.score) for result in found.results)
    ranked_ids = [chunk_id for chunk_id, _ in ranking]
    outcome = QueryOutcome(
        query_id=query.query_id,
        direction=direction_of(query, kb.passages),
        ranks=_ranks_of(kb, ranked_ids, query.relevant),
    )
    return QueryResult(
        query=query,
        ranking=ranking,
        outcome=outcome,
        secondary_ranks=_ranks_of(kb, ranked_ids, query.secondary),
        truncation_group=truncation_group(kb, provider, query),
        search_seconds=search_seconds,
    )


def evaluate(kb: KnowledgeBase, provider: TokenCountingProvider,
             queries: Sequence[GoldenQuery] = QUERIES) -> tuple[QueryResult, ...]:
    """Every golden question, in dataset order."""
    return tuple(evaluate_query(kb, provider, query) for query in queries)


def relevant_chunk_ids(kb: KnowledgeBase, query: GoldenQuery) -> set[str]:
    """Chunk IDs that contain any relevant evidence of the query."""
    return {chunk.chunk_id for evidence in query.relevant
            for chunk in evidence_chunks(kb, evidence)}
