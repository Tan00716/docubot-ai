"""Batch 6A regressions: query token limit, "query: " prefix, full-precision ranking.

FastEmbedProvider is tested with a stub model object in place of the loaded
ONNX model, so its real code paths (token check, prefix, count_tokens) run
without downloading anything. The real tokenizer is exercised in
test_retrieval_evaluation_model.py (opt-in).

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import logging
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from app import api
from app.document_processing.models import Chunk
from app.embeddings.config import EmbeddingConfig
from app.embeddings.models import QueryTooLongError
from app.embeddings.provider import FastEmbedProvider
from app.search import service
from app.search.models import SCORE_DECIMALS, SearchResult
from fake_embeddings import FakeEmbeddingProvider
from test_search import SearchTestCase, unit
from test_search_api import SearchApiTestCase

logging.getLogger("app").setLevel(logging.CRITICAL)

DIMENSION = 384
QUERY_TOO_LONG_MESSAGE = QueryTooLongError.safe_message
QUESTIONS = {
    "en": "How does exact vector search rank chunks?",
    "zh": "向量检索是怎样给文本块排序的？",
    "mixed": "exact search 的 ranking 是怎么算的？",
}


class StubTokenizer:
    """Pretends one whitespace-separated word is one token, plus 2 special tokens."""

    def __init__(self, max_tokens=512):
        self.max_tokens = max_tokens
        self.inputs = []

    def encode(self, text):
        self.inputs.append(text)
        too_long = len(text.split()) + 2 > self.max_tokens
        return SimpleNamespace(overflowing=[SimpleNamespace()] if too_long else [])


class StubModel:
    """Stands in for the loaded fastembed TextEmbedding object."""

    def __init__(self, tokenizer):
        self.model = SimpleNamespace(tokenizer=tokenizer)
        self.embedded = []

    def embed(self, inputs, batch_size):
        self.embedded.extend(inputs)
        return iter([np.ones(DIMENSION) for _ in inputs])


def stub_provider(tokenizer=None):
    provider = FastEmbedProvider(EmbeddingConfig(cache_dir=Path("unused-cache")))
    provider._model = StubModel(tokenizer or StubTokenizer())  # "already loaded"
    return provider


def words(count):
    return " ".join(["word"] * count)


# --- Query token limit (real provider code, stub model) -------------------------


class QueryTokenLimitTests(unittest.TestCase):
    def test_short_query_is_embedded(self):
        provider = stub_provider()

        vector = provider.embed_query("What is FastAPI?")

        self.assertEqual(len(vector), DIMENSION)
        self.assertEqual(provider._model.embedded, ["query: What is FastAPI?"])

    def test_query_exactly_at_the_limit_is_embedded(self):
        provider = stub_provider()
        # "query:" + 509 words + 2 special tokens = 512 tokens.
        provider.embed_query(words(509))

        self.assertEqual(len(provider._model.embedded), 1)

    def test_query_one_token_over_the_limit_is_rejected_before_embedding(self):
        provider = stub_provider()

        with self.assertRaises(QueryTooLongError):
            provider.embed_query(words(510))
        self.assertEqual(provider._model.embedded, [], "the model must not embed a cut question")

    def test_the_limit_is_measured_on_the_exact_model_input(self):
        tokenizer = StubTokenizer()
        provider = stub_provider(tokenizer)

        provider.embed_query("Hello")

        self.assertEqual(tokenizer.inputs, ["query: Hello"])

    def test_documents_are_still_embedded_and_marked_truncated(self):
        # The policy is for questions only: long chunks keep truncated=True.
        provider = stub_provider(StubTokenizer(max_tokens=5))

        [embedded] = provider.embed_documents([words(20)])

        self.assertTrue(embedded.truncated)


# --- "query: " prefix for every language ---------------------------------------


class QueryPrefixTests(unittest.TestCase):
    def test_every_language_gets_the_query_prefix_never_the_passage_prefix(self):
        for language, question in QUESTIONS.items():
            with self.subTest(language=language):
                provider = stub_provider()

                provider.embed_query(question)

                self.assertEqual(provider._model.embedded, ["query: " + question])
                self.assertFalse(provider._model.embedded[0].startswith("passage: "))

    def test_search_passes_the_bare_question_and_the_provider_adds_the_prefix(self):
        provider = FakeEmbeddingProvider()
        for question in QUESTIONS.values():
            provider.embed_query(question)

        self.assertEqual(provider.model_inputs, ["query: " + q for q in QUESTIONS.values()])


# --- count_tokens: exact length before the cut, no second model load -----------


class CountTokensTests(unittest.TestCase):
    def make_tokenizer(self):
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.enable_truncation(max_length=4)
        return tokenizer

    def test_counts_all_tokens_even_beyond_the_models_cut(self):
        tokenizer = self.make_tokenizer()
        provider = stub_provider(tokenizer)

        with patch.object(provider, "_load_model") as load:
            count = provider.count_tokens("passage: a a a a a a a")
        load.assert_not_called()

        self.assertEqual(count, 8)
        # The model's own tokenizer is untouched and still cuts at 4.
        self.assertEqual(len(tokenizer.encode("passage: a a a a a a a").ids), 4)

    def test_the_counting_copy_is_made_only_once(self):
        provider = stub_provider(self.make_tokenizer())

        provider.count_tokens("a a")
        first = provider._counting_tokenizer
        provider.count_tokens("a a a")

        self.assertIs(provider._counting_tokenizer, first)


# --- Full-precision ranking through the search service -------------------------


class FullPrecisionSearchTests(SearchTestCase):
    def two_close_chunks(self):
        """Two chunks whose scores both display as 0.912345 but differ at 1e-7."""
        chunks = sorted([self.one_chunk_document("First close chunk."),
                         self.one_chunk_document("Second close chunk.")],
                        key=lambda c: c.chunk_id)
        low, high = 0.91234512, 0.91234549
        # The HIGHER score goes to the LARGER chunk_id: with rounding before
        # sorting, the tie-breaker (chunk_id ASC) would put it second.
        self.store_vector(chunks[0], unit(low, (1 - low ** 2) ** 0.5))
        self.store_vector(chunks[1], unit(high, (1 - high ** 2) ** 0.5))
        return chunks

    def test_close_scores_are_ranked_by_their_exact_value(self):
        smaller_id, larger_id = self.two_close_chunks()

        results = self.run_search(top_k=2).results

        self.assertEqual([r.chunk.chunk_id for r in results],
                         [larger_id.chunk_id, smaller_id.chunk_id])
        self.assertEqual(round(results[0].score, 6), round(results[1].score, 6))
        self.assertGreater(results[0].score, results[1].score)

    def test_service_results_keep_the_unrounded_score(self):
        self.two_close_chunks()

        [best, _] = self.run_search(top_k=2).results

        self.assertNotEqual(best.score, round(best.score, SCORE_DECIMALS))

    def test_query_too_long_propagates_from_the_service(self):
        self.two_close_chunks()
        provider = FakeEmbeddingProvider()

        with self.assertRaises(QueryTooLongError):
            self.run_search(query=words(510), provider=provider)

    def test_over_limit_query_with_nothing_to_search_returns_no_results(self):
        # Documented policy: the question is only measured when it is embedded,
        # and with no candidates it is never embedded (the model is not loaded).
        results = self.run_search(query=words(510), provider=FakeEmbeddingProvider())

        self.assertEqual(results.results, ())


# --- API: rounding only for display, 422 for over-limit questions --------------


class SearchApiPrecisionAndLimitTests(SearchApiTestCase):
    def test_response_rounds_the_score_for_display_only(self):
        chunk = Chunk(chunk_id="c" * 64, document_id="d" * 32, chunk_index=0, text="text",
                      char_count=4, source_filename="a.txt", file_type="txt",
                      source_locations=({"block": 0},), chunking_version=1,
                      chunk_size=1200, chunk_overlap=200)
        result = SearchResult(rank=1, score=0.123456789, chunk=chunk, truncated=False)

        self.assertEqual(api.SearchResultResponse.from_result(result).score, 0.123457)

    def test_over_limit_query_returns_422_with_a_safe_message(self):
        self.embedded_document()
        self.provider.model_inputs.clear()

        response = self.search(query=words(510))

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json(), {"detail": QUERY_TOO_LONG_MESSAGE})
        self.assertNotIn("word word", response.text, "the question is not echoed back")
        self.assertEqual(self.provider.model_inputs, [])

    def test_near_limit_query_is_searched(self):
        self.embedded_document()

        # "query:" + 509 words + 2 special tokens = exactly 512 fake tokens.
        response = self.search(query=words(509), top_k=1)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 1)

    def test_character_limit_still_applies_before_any_token_check(self):
        self.embedded_document()
        self.provider.model_inputs.clear()

        response = self.search(query="x" * (api.MAX_QUERY_LENGTH + 1))

        self.assert_validation_error(response)


if __name__ == "__main__":
    unittest.main()
