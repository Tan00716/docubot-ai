"""Retrieval evaluation with the REAL local model (opt-in, not part of the normal run).

Builds the golden corpus twice at the production chunking (1200/200), runs
all golden questions through app.search.service.search, and checks:

- the query token limit with the real tokenizer (512 accepted, 513 rejected)
- the stored truncated flag agrees with the real token count of every chunk
- Chinese, English and mixed questions are embedded with "query: "
- every language direction is evaluated and reported
- two independent builds give identical rankings and full-precision scores
- regression floors (set BELOW the measured values, see README "Retrieval
  Evaluation"); they catch a broken pipeline, they are not quality targets

Run it explicitly from the project root (PowerShell):

    $env:DOCUBOT_RUN_MODEL_SMOKE_TEST = "1"
    .venv\\Scripts\\python.exe -m unittest discover -s tests -p "test_retrieval_evaluation_model.py" -v
"""

import logging
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import api
from app.document_processing.chunking import ChunkingConfig
from app.embeddings.config import EmbeddingConfig
from app.embeddings.models import QueryTooLongError
from app.embeddings.provider import FastEmbedProvider
from app.embeddings.vectors import l2_normalize, to_float32
from app.evaluation import analysis, runner
from app.evaluation.golden import DIRECTIONS
from app.evaluation.metrics import summarize, summarize_groups
from app.search import service
from test_embedding_smoke import model_is_cached

logging.getLogger("app").setLevel(logging.CRITICAL)

QUESTIONS = {
    "en": "How does exact vector search rank chunks?",
    "zh": "向量检索是怎样给文本块排序的？",
    "mixed": "exact search 的 ranking 是怎么算的？",
}
# Regression floors, below the values measured on 2026-09-25 (see README).
MIN_OVERALL_RECALL_AT_5 = 0.55  # measured 0.667
MIN_SAME_LANGUAGE_RECALL_AT_5 = 0.80  # en->en + zh->zh, measured 0.929


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


@unittest.skipUnless(os.environ.get("DOCUBOT_RUN_MODEL_SMOKE_TEST") == "1",
                     "set DOCUBOT_RUN_MODEL_SMOKE_TEST=1 to run the real model evaluation")
class RealModelEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cache_dir = api.MODEL_CACHE_DIR
        cls.offline = {"HF_HUB_OFFLINE": "1"} if model_is_cached(cache_dir) else {}
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        with patch.dict(os.environ, cls.offline):
            cls.provider = FastEmbedProvider(EmbeddingConfig(cache_dir=cache_dir))
            cls.kb = runner.build_knowledge_base(root / "a", cls.provider,
                                                 ChunkingConfig(1200, 200))
            cls.results = runner.evaluate(cls.kb, cls.provider)
            cls.second_kb = runner.build_knowledge_base(root / "b", cls.provider,
                                                        ChunkingConfig(1200, 200))
            cls.second_results = runner.evaluate(cls.second_kb, cls.provider)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def raw(self, model_input):
        model = self.provider._get_model()
        return to_float32(l2_normalize(next(iter(model.embed([model_input], batch_size=1)))
                                       .tolist()))

    def test_query_token_limit_with_the_real_tokenizer(self):
        count = lambda n: self.provider.count_tokens("query: " + " ".join(["word"] * n))
        at_limit = next(n for n in range(300, 600) if count(n) >= 512)
        self.assertEqual(count(at_limit), 512, "one word = one token in this sentence")

        self.assertEqual(len(self.provider.embed_query(" ".join(["word"] * at_limit))), 384)
        with self.assertRaises(QueryTooLongError):
            self.provider.embed_query(" ".join(["word"] * (at_limit + 1)))

    def test_long_chinese_question_under_the_character_limit_is_rejected(self):
        question = "向量检索怎样处理很长的问题？" * 60  # 840 characters, far below 4000

        self.assertGreater(self.provider.count_tokens("query: " + question), 512)
        with self.assertRaises(QueryTooLongError):
            service.search(question, db_path=self.kb.db_path, provider=self.provider)

    def test_stored_truncated_flag_matches_the_real_token_count(self):
        for chunk in self.kb.chunks.values():
            with self.subTest(chunk=chunk.passage_key):
                self.assertEqual(chunk.truncated, chunk.tokens > 512)
        stats = analysis.chunk_stats(self.kb)
        self.assertGreater(stats.truncated_by_language["zh"], 0)
        self.assertEqual(stats.truncated_by_language["en"], 0)

    def test_every_language_uses_the_query_prefix(self):
        for language, question in QUESTIONS.items():
            with self.subTest(language=language):
                vector = self.provider.embed_query(question)
                self.assertAlmostEqual(dot(vector, self.raw("query: " + question)), 1.0, places=5)
                self.assertLess(dot(vector, self.raw("passage: " + question)), 1.0 - 1e-4)
                self.assertLess(dot(vector, self.raw(question)), 1.0 - 1e-4)

    def test_every_direction_is_evaluated_and_floors_hold(self):
        outcomes = [r.outcome for r in self.results]
        overall = summarize(outcomes)
        groups = summarize_groups(outcomes, lambda o: o.direction, DIRECTIONS)
        same = summarize([o for o in outcomes if o.direction in ("en->en", "zh->zh")])
        lines = [f"  {name:13} n={s.queries:2} R@1={s.recall[1]:.3f} R@5={s.recall[5]:.3f} "
                 f"MRR={s.mrr:.3f}" for name, s in groups.items()]
        print("\n  Retrieval evaluation (development set):\n" + "\n".join(lines)
              + f"\n  all           n={overall.queries} R@1={overall.recall[1]:.3f} "
              f"R@5={overall.recall[5]:.3f} MRR={overall.mrr:.3f}", file=sys.stderr)

        self.assertTrue(all(summary is not None for summary in groups.values()))
        self.assertGreaterEqual(overall.recall[5], MIN_OVERALL_RECALL_AT_5)
        self.assertGreaterEqual(same.recall[5], MIN_SAME_LANGUAGE_RECALL_AT_5)

    def test_two_independent_builds_give_identical_results(self):
        self.assertEqual(analysis.fingerprint(self.results),
                         analysis.fingerprint(self.second_results))
        self.assertEqual(summarize([r.outcome for r in self.results]),
                         summarize([r.outcome for r in self.second_results]))


if __name__ == "__main__":
    unittest.main()
