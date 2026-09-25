"""Tests for the retrieval evaluation tooling in app/evaluation/ (no real model).

Metrics are tested as pure functions. The runner is tested end to end with
the fast fake provider: it still uses the real upload -> process -> chunk ->
embed -> search pipeline, only the vectors are fake. The real-model
evaluation lives in test_retrieval_evaluation_model.py (opt-in).

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import logging
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from app.document_processing.chunking import ChunkingConfig
from app.evaluation import analysis, runner
from app.evaluation.corpus import LANGUAGES, PASSAGES, Passage
from app.evaluation.golden import (
    DIRECTIONS,
    QUERIES,
    Evidence,
    GoldenQuery,
    direction_of,
    validate_dataset,
)
from app.evaluation.metrics import (
    QueryOutcome,
    item_ranks,
    recall_at_k,
    reciprocal_rank,
    summarize,
    summarize_groups,
)
from fake_embeddings import FakeEmbeddingProvider

logging.getLogger("app").setLevel(logging.CRITICAL)

MAIN_DIRECTIONS = ("en->en", "zh->zh", "zh->en", "en->zh")
MIXED_DIRECTIONS = ("mixed->en", "mixed->zh")
# A question may share a product name or a term with its answer ("long
# polling", "FastAPI"), but never a copied clause. Measured maximum: 13.
MAX_SHARED_PHRASE = 16


def outcome(ranks, direction="en->en", query_id="q"):
    return QueryOutcome(query_id=query_id, direction=direction, ranks=tuple(ranks))


def longest_common_substring(a: str, b: str) -> int:
    best, previous = 0, [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


class CharTokenProvider(FakeEmbeddingProvider):
    """Fake provider where one character is one token, and truncation agrees with it.

    The model "reads" at most max_tokens characters of "passage: " + text, so
    a chunk is truncated exactly when its model input is longer than that.
    """

    def __init__(self, max_tokens=60, **options):
        prefix = len("passage: ")
        super().__init__(max_tokens=max_tokens, truncate_after=max_tokens - prefix, **options)

    def count_tokens(self, model_input):
        return len(model_input)


class AimedProvider(CharTokenProvider):
    """Its query vector IS the passage vector of one chosen text: that chunk scores 1.0."""

    def __init__(self, target_text, **options):
        super().__init__(**options)
        self.target_text = target_text

    def embed_query(self, text):
        super().embed_query(text)
        return self.passage_vector(self.target_text)


def temporary_root(test):
    folder = tempfile.TemporaryDirectory()
    test.addCleanup(folder.cleanup)
    return Path(folder.name)


# --- Metrics ------------------------------------------------------------------


class RecallTests(unittest.TestCase):
    def test_single_item_is_found_only_when_inside_the_top_k(self):
        self.assertEqual(recall_at_k([3], 1), 0.0)
        self.assertEqual(recall_at_k([3], 3), 1.0)
        self.assertEqual(recall_at_k([None], 5), 0.0)

    def test_recall_counts_relevant_items_not_one_answer(self):
        # Two relevant items, only one in the top 3: half of the evidence found.
        self.assertEqual(recall_at_k([2, 9], 3), 0.5)
        self.assertEqual(recall_at_k([2, None], 5), 0.5)
        self.assertEqual(recall_at_k([1, 2], 1), 0.5)

    def test_recall_never_decreases_with_larger_k(self):
        for ranks in ([4], [1, 7], [None, 2], [5, 5, None]):
            with self.subTest(ranks=ranks):
                values = [recall_at_k(ranks, k) for k in range(1, 11)]
                self.assertEqual(values, sorted(values))

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_k([], 1)
        with self.assertRaises(ValueError):
            recall_at_k([1], 0)


class ItemRankTests(unittest.TestCase):
    def test_an_item_ranks_where_its_best_chunk_ranks(self):
        # Overlap copied the evidence into chunks c2 and c4: the item ranks 2.
        self.assertEqual(item_ranks(["c1", "c2", "c3", "c4"], [{"c4", "c2"}]), (2,))

    def test_evidence_in_two_chunks_is_still_one_item(self):
        ranks = item_ranks(["c2", "c4", "x"], [{"c2", "c4"}])

        self.assertEqual(ranks, (1,))
        self.assertEqual(recall_at_k(ranks, 1), 1.0)

    def test_unreturned_items_have_no_rank(self):
        self.assertEqual(item_ranks(["a", "b"], [{"z"}, {"b"}]), (None, 2))

    def test_duplicate_chunks_in_a_ranking_are_rejected(self):
        with self.assertRaises(ValueError):
            item_ranks(["a", "a"], [{"a"}])


class ReciprocalRankTests(unittest.TestCase):
    def test_values_for_rank_1_2_4_and_missing(self):
        self.assertEqual(reciprocal_rank([1]), 1.0)
        self.assertEqual(reciprocal_rank([2]), 0.5)
        self.assertEqual(reciprocal_rank([4]), 0.25)
        self.assertEqual(reciprocal_rank([None]), 0.0)

    def test_the_best_ranked_item_decides(self):
        self.assertEqual(reciprocal_rank([None, 5, 2]), 0.5)

    def test_invalid_ranks_are_rejected(self):
        with self.assertRaises(ValueError):
            reciprocal_rank([])
        with self.assertRaises(ValueError):
            reciprocal_rank([0])


class SummaryTests(unittest.TestCase):
    def test_mrr_and_recall_are_averages_over_queries(self):
        summary = summarize([outcome([1]), outcome([2]), outcome([4]), outcome([None])])

        self.assertEqual(summary.queries, 4)
        self.assertAlmostEqual(summary.mrr, (1 + 0.5 + 0.25 + 0) / 4)
        self.assertEqual(summary.recall, {1: 0.25, 3: 0.5, 5: 0.75})

    def test_every_query_counts_once_whatever_its_number_of_items(self):
        summary = summarize([outcome([1, None]), outcome([None])])

        self.assertEqual(summary.recall[1], (0.5 + 0.0) / 2)

    def test_groups_keep_every_category_even_without_queries(self):
        groups = summarize_groups([outcome([1], "zh->en")], lambda o: o.direction, DIRECTIONS)

        self.assertEqual(list(groups), list(DIRECTIONS))
        self.assertEqual(groups["zh->en"].queries, 1)
        self.assertIsNone(groups["mixed->zh"])

    def test_unknown_group_is_an_error_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            summarize_groups([outcome([1], "xx->yy")], lambda o: o.direction, DIRECTIONS)

    def test_empty_summary_is_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])


# --- Golden dataset -----------------------------------------------------------


class GoldenDatasetTests(unittest.TestCase):
    def test_dataset_is_valid(self):
        validate_dataset()

    def test_corpus_size_and_language_mix(self):
        languages = Counter(passage.language for passage in PASSAGES)

        self.assertTrue(30 <= len(PASSAGES) <= 50, len(PASSAGES))
        self.assertTrue(20 <= len(QUERIES) <= 35, len(QUERIES))
        self.assertEqual(set(languages), set(LANGUAGES))
        self.assertTrue(all(languages[language] >= 5 for language in LANGUAGES), languages)

    def test_every_required_direction_is_covered(self):
        counts = Counter(direction_of(query) for query in QUERIES)

        for direction in MAIN_DIRECTIONS:
            self.assertGreaterEqual(counts[direction], 5, direction)
        for direction in MIXED_DIRECTIONS:
            self.assertGreaterEqual(counts[direction], 3, direction)
        self.assertEqual(set(counts) - set(DIRECTIONS), set())

    def test_questions_do_not_copy_long_phrases_from_their_answer(self):
        text_of = {passage.key: passage.text.lower() for passage in PASSAGES}
        for query in QUERIES:
            for evidence in query.relevant:
                with self.subTest(query=query.query_id):
                    shared = longest_common_substring(query.text.lower(),
                                                      text_of[evidence.passage_key])
                    self.assertLessEqual(shared, MAX_SHARED_PHRASE)

    def test_direction_is_question_language_to_answer_passage_language(self):
        query = GoldenQuery("q", "zh", "问题", (Evidence("en_postgres", "PostgreSQL"),))

        self.assertEqual(direction_of(query), "zh->en")

    def test_evidence_that_is_not_in_its_passage_is_rejected(self):
        broken = GoldenQuery("q", "en", "question", (Evidence("en_flask", "not in the text"),))

        with self.assertRaises(ValueError):
            validate_dataset(PASSAGES, (broken,))

    def test_a_passage_cannot_be_relevant_and_secondary(self):
        both = GoldenQuery("q", "en", "question", (Evidence("en_flask", "Flask"),),
                           (Evidence("en_flask", "routing"),))

        with self.assertRaises(ValueError):
            validate_dataset(PASSAGES, (both,))

    def test_duplicate_query_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_dataset(PASSAGES, (QUERIES[0], QUERIES[0]))


# --- Runner with the real pipeline and a fake model ------------------------------


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Two independent builds of the whole corpus, shared by the read-only
        # tests below (building is the slow part).
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        cls.provider = FakeEmbeddingProvider()
        cls.kb = runner.build_knowledge_base(root / "a", cls.provider, ChunkingConfig(1200, 200))
        cls.second_kb = runner.build_knowledge_base(root / "b", cls.provider,
                                                    ChunkingConfig(1200, 200))

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def build(self, provider, config, passages):
        return provider, runner.build_knowledge_base(temporary_root(self), provider, config,
                                                     passages)

    def test_every_passage_becomes_chunks_and_every_label_finds_its_chunks(self):
        kb = self.kb

        self.assertEqual({c.passage_key for c in kb.chunks.values()},
                         {p.key for p in PASSAGES})
        self.assertEqual(len(kb.chunks), len(PASSAGES), "at 1200/200 every passage is one chunk")
        for query in QUERIES:
            for evidence in (*query.relevant, *query.secondary):
                self.assertTrue(runner.evidence_chunks(kb, evidence))

    def test_ranks_come_from_the_search_results(self):
        provider, kb = self.provider, self.kb

        for result in runner.evaluate(kb, provider):
            with self.subTest(query=result.query.query_id):
                ranked_ids = [chunk_id for chunk_id, _ in result.ranking]
                relevant = runner.relevant_chunk_ids(kb, result.query)
                expected = min((i for i, c in enumerate(ranked_ids, 1) if c in relevant),
                               default=None)
                self.assertEqual(min((r for r in result.outcome.ranks if r), default=None),
                                 expected)
                scores = [score for _, score in result.ranking]
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_document_ids_are_stable_so_rankings_are_reproducible(self):
        self.assertEqual(runner.passage_document_id("zh_faq"),
                         runner.passage_document_id("zh_faq"))
        self.assertNotEqual(runner.passage_document_id("zh_faq"),
                            runner.passage_document_id("en_flask"))
        self.assertEqual(list(self.kb.chunks), list(self.second_kb.chunks))

    def test_two_independent_evaluations_are_identical(self):
        first = runner.evaluate(self.kb, self.provider)
        second = runner.evaluate(self.second_kb, self.provider)

        self.assertEqual(analysis.fingerprint(first), analysis.fingerprint(second))
        self.assertEqual(summarize([r.outcome for r in first]),
                         summarize([r.outcome for r in second]))

    def test_smaller_chunks_split_long_passages_and_keep_labels_usable(self):
        provider, kb = self.build(FakeEmbeddingProvider(), ChunkingConfig(400, 50), PASSAGES)

        self.assertGreater(len(kb.chunks), len(PASSAGES))
        stats = analysis.chunk_stats(kb)
        self.assertEqual(stats.chunks, len(kb.chunks))
        self.assertLessEqual(max(len(c.text) for c in kb.chunks.values()), 400)
        self.assertAlmostEqual(stats.average_chars,
                               sum(len(c.text) for c in kb.chunks.values()) / len(kb.chunks))
        self.assertEqual(sum(stats.chunks_by_language.values()), stats.chunks)
        for query in QUERIES:
            runner.relevant_chunk_ids(kb, query)  # raises if a label lost its chunk

    def test_evidence_split_across_a_chunk_boundary_is_reported(self):
        words = " ".join(f"w{i}" for i in range(200))
        passage = Passage("long", "en", words)
        _, kb = self.build(CharTokenProvider(max_tokens=4000), ChunkingConfig(300, 0),
                           [passage])
        cut = kb.chunks[next(iter(kb.chunks))].text.split()[-1]  # last word of a chunk

        with self.assertRaises(runner.DatasetError):
            runner.evidence_chunks(kb, Evidence("long", f"{cut} w"))

    def test_secondary_evidence_never_counts_as_relevant(self):
        relevant = Passage("relevant", "en", "The relevant passage about caching layers.")
        secondary = Passage("secondary", "en", "A partly related passage about layers.")
        query = GoldenQuery("q", "en", "question?", (Evidence("relevant", "caching layers"),),
                            (Evidence("secondary", "related passage"),))
        provider, kb = self.build(AimedProvider(secondary.text), ChunkingConfig(1200, 200),
                                  [relevant, secondary])

        [result] = runner.evaluate(kb, provider, [query])

        self.assertEqual((result.outcome.ranks, result.secondary_ranks), ((2,), (1,)))
        self.assertEqual(recall_at_k(result.outcome.ranks, 1), 0.0)


class TruncationTests(unittest.TestCase):
    """CharTokenProvider: the model reads 60 characters of "passage: " + text."""

    def setUp(self):
        self.provider = CharTokenProvider(max_tokens=60)
        self.passages = [
            Passage("short", "en", "A short passage about queues."),
            Passage("long", "en", "Early fact about ports. " + "Filler words go here. " * 4
                    + "Late fact about sockets."),
        ]
        self.kb = runner.build_knowledge_base(temporary_root(self), self.provider,
                                              ChunkingConfig(1200, 200), self.passages)

    def query(self, key, evidence):
        return GoldenQuery("q", "en", "question?", (Evidence(key, evidence),))

    def test_truncated_flag_is_the_stored_embedding_metadata(self):
        by_key = {c.passage_key: c for c in self.kb.chunks.values()}

        self.assertEqual((by_key["short"].truncated, by_key["long"].truncated), (False, True))
        self.assertEqual(by_key["long"].tokens, len("passage: " + by_key["long"].text))

    def test_queries_are_grouped_by_what_the_model_read(self):
        groups = [runner.truncation_group(self.kb, self.provider, self.query(key, evidence))
                  for key, evidence in [("short", "about queues"),
                                        ("long", "Early fact about ports"),
                                        ("long", "Late fact about sockets")]]

        self.assertEqual(groups, list(runner.TRUNCATION_GROUPS))

    def test_token_slices_and_limits_use_the_provider_tokenizer(self):
        slices = analysis.token_slices(self.provider, self.passages, lengths=(20, 60))

        self.assertEqual([(s.language, s.chars) for s in slices], [("en", 20), ("en", 60)])
        self.assertEqual([s.tokens for s in slices], [29, 69])
        self.assertEqual([s.truncated for s in slices], [False, True])
        self.assertEqual(analysis.max_chars_within_limit(self.provider, "x" * 100),
                         60 - len("passage: "))
        self.assertEqual(analysis.max_chars_within_limit(self.provider, "x" * 10), 10)

    def test_diagnosis_reports_rank_score_and_competitor(self):
        provider = AimedProvider(self.passages[0].text, max_tokens=60)
        kb = runner.build_knowledge_base(temporary_root(self), provider,
                                         ChunkingConfig(1200, 200), self.passages)
        [result] = runner.evaluate(kb, provider, [self.query("long", "Early fact about ports")])

        diagnosis = analysis.diagnose(kb, result)

        self.assertEqual(diagnosis.first_relevant_rank, 2)
        self.assertEqual(diagnosis.competitor_language, "en")
        self.assertAlmostEqual(diagnosis.competitor_score, 1.0, places=5)
        self.assertEqual(diagnosis.query_language_in_top, 2)


if __name__ == "__main__":
    unittest.main()
