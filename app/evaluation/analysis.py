"""Measurements behind the report: tokens, chunk statistics, per-query diagnosis.

Token numbers always come from the embedding model's own tokenizer
(provider.count_tokens), never from a characters-per-token guess, and are
measured separately for English, Chinese and mixed text.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.corpus import LANGUAGES, PASSAGES, Language, Passage
from app.evaluation.runner import KnowledgeBase, QueryResult, TokenCountingProvider, relevant_chunk_ids

SLICE_LENGTHS = (600, 800, 1000, 1200)  # characters
TOP_N_FOR_LANGUAGE_MIX = 5


@dataclass(frozen=True)
class TokenSlice:
    language: Language
    chars: int
    tokens: int  # of "passage: " + the first `chars` characters, special tokens included
    truncated: bool  # tokens > the model's limit


def language_text(language: Language, passages: Sequence[Passage] = PASSAGES) -> str:
    """All corpus passages of one language, joined like blocks of one document."""
    return "\n\n".join(passage.text for passage in passages if passage.language == language)


def token_slices(provider: TokenCountingProvider, passages: Sequence[Passage] = PASSAGES,
                 lengths: Sequence[int] = SLICE_LENGTHS) -> list[TokenSlice]:
    """Token count of the first N characters of each language's real corpus text."""
    limit = provider.contract.max_tokens
    slices = []
    for language in LANGUAGES:
        text = language_text(language, passages)
        for length in lengths:
            if len(text) < length:
                continue
            tokens = provider.count_tokens(provider.contract.passage_input(text[:length]))
            slices.append(TokenSlice(language, length, tokens, tokens > limit))
    return slices


def max_chars_within_limit(provider: TokenCountingProvider, text: str) -> int:
    """The longest prefix of `text` (in characters) that the model reads completely.

    Binary search over prefix lengths; a longer prefix never has fewer tokens.
    """
    limit = provider.contract.max_tokens

    def fits(length: int) -> bool:
        return provider.count_tokens(provider.contract.passage_input(text[:length])) <= limit

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low


@dataclass(frozen=True)
class ChunkStats:
    chunk_size: int
    chunk_overlap: int
    chunks: int
    average_chars: float
    max_tokens: int
    truncated: int
    chunks_by_language: dict[str, int]
    truncated_by_language: dict[str, int]


def chunk_stats(kb: KnowledgeBase) -> ChunkStats:
    chunks = list(kb.chunks.values())
    if not chunks:
        raise ValueError("The knowledge base has no chunks.")
    return ChunkStats(
        chunk_size=kb.config.chunk_size,
        chunk_overlap=kb.config.chunk_overlap,
        chunks=len(chunks),
        average_chars=sum(len(c.text) for c in chunks) / len(chunks),
        max_tokens=max(c.tokens for c in chunks),
        truncated=sum(c.truncated for c in chunks),
        chunks_by_language={lang: sum(c.language == lang for c in chunks) for lang in LANGUAGES},
        truncated_by_language={lang: sum(c.truncated and c.language == lang for c in chunks)
                               for lang in LANGUAGES},
    )


@dataclass(frozen=True)
class QueryDiagnosis:
    query_id: str
    direction: str
    first_relevant_rank: int | None
    first_relevant_score: float | None
    competitor_language: str | None  # best-ranked chunk that is neither relevant nor secondary
    competitor_score: float | None
    query_language_in_top: int  # top-N results written in the question's language


def diagnose(kb: KnowledgeBase, result: QueryResult) -> QueryDiagnosis:
    relevant = relevant_chunk_ids(kb, result.query)
    secondary_keys = {evidence.passage_key for evidence in result.query.secondary}
    first_rank = first_score = competitor_language = competitor_score = None
    for rank, (chunk_id, score) in enumerate(result.ranking, start=1):
        chunk = kb.chunks[chunk_id]
        if chunk_id in relevant and first_rank is None:
            first_rank, first_score = rank, score
        is_other = chunk_id not in relevant and chunk.passage_key not in secondary_keys
        if is_other and competitor_language is None:
            competitor_language, competitor_score = chunk.language, score
    top = result.ranking[:TOP_N_FOR_LANGUAGE_MIX]
    same_language = sum(kb.chunks[chunk_id].language == result.query.language
                        for chunk_id, _ in top)
    return QueryDiagnosis(
        query_id=result.query.query_id,
        direction=result.outcome.direction,
        first_relevant_rank=first_rank,
        first_relevant_score=first_score,
        competitor_language=competitor_language,
        competitor_score=competitor_score,
        query_language_in_top=same_language,
    )


def fingerprint(results: Sequence[QueryResult]) -> tuple:
    """Everything that must be identical between two runs: rankings and exact scores."""
    return tuple((r.query.query_id, r.ranking, r.outcome.ranks, r.secondary_ranks,
                  r.truncation_group) for r in results)
