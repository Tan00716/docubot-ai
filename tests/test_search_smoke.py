"""Vector search with the REAL local model (opt-in, not part of the normal run).

Two parts:

1. Smoke test: a 4-document corpus and 4 questions. Checks that the whole
   pipeline (passage embeddings -> SQLite -> query embedding -> exact
   search) works with intfloat/multilingual-e5-small, that the scores are
   exactly dot("query: " + question, "passage: " + chunk) of the raw model,
   and that an obviously related chunk ranks above clearly unrelated ones.
   It proves the CONTRACT and the pipeline, not retrieval quality.

2. Retrieval baseline: a small hand-labelled set (tests/retrieval_baseline.py)
   and Recall@1 / @3 / @5. A development baseline for later comparison, not
   a benchmark and not a production quality claim.

Run it explicitly from the project root (PowerShell):

    $env:DOCUBOT_RUN_MODEL_SMOKE_TEST = "1"
    .venv\\Scripts\\python.exe -m unittest discover -s tests -p "test_search_smoke.py" -v
"""

import math
import os
import sys
import time
import unittest
from unittest.mock import patch

from app import api
from app.document_processing.chunking import ChunkingConfig
from app.embeddings import service as embedding_service
from app.embeddings.config import EmbeddingConfig
from app.embeddings.provider import FastEmbedProvider
from app.embeddings.vectors import l2_normalize, to_float32
from app.search import service
from retrieval_baseline import PASSAGES, QUERIES, recall_at_k
from test_embedding_smoke import model_is_cached
from test_embedding_storage import EmbeddingTestCase

SMOKE_CORPUS = {
    "fastapi": "FastAPI is a Python web framework for building APIs.",
    "telegram": "Telegram bots can receive and send messages.",
    "sqlite": "SQLite is an embedded relational database.",
    "embeddings": "Vector embeddings represent text as numerical vectors.",
}
SMOKE_QUERIES = {
    "What is FastAPI used for?": "fastapi",
    "How do Telegram bots work?": "telegram",
    "What is SQLite?": "sqlite",
    "How are text embeddings represented?": "embeddings",
}
# Regression floors for the baseline (set BELOW the measured values, see
# README "Retrieval baseline"). They catch a broken pipeline, e.g. swapped
# prefixes; they are not quality targets.
MIN_RECALL = {1: 0.6, 3: 0.8, 5: 0.9}


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


@unittest.skipUnless(os.environ.get("DOCUBOT_RUN_MODEL_SMOKE_TEST") == "1",
                     "set DOCUBOT_RUN_MODEL_SMOKE_TEST=1 to run the real model search tests")
class RealModelSearchTest(EmbeddingTestCase):
    @classmethod
    def setUpClass(cls):
        cache_dir = api.MODEL_CACHE_DIR
        # With a cached model, network access to Hugging Face is switched off.
        cls.offline = {"HF_HUB_OFFLINE": "1"} if model_is_cached(cache_dir) else {}
        with patch.dict(os.environ, cls.offline):
            cls.provider = FastEmbedProvider(EmbeddingConfig(cache_dir=cache_dir))
            cls.raw_model = cls.provider._get_model()

    def raw(self, model_input):
        values = next(iter(self.raw_model.embed([model_input], batch_size=1))).tolist()
        return to_float32(l2_normalize(values))

    def build_corpus(self, passages):
        """One embedded one-chunk document per passage; returns chunk_id -> key."""
        keys = {}
        for key, text in passages.items():
            document_id = self.chunked_document(text.encode("utf-8"), ChunkingConfig(1200, 200))
            [chunk] = self.chunks(document_id)
            keys[chunk.chunk_id] = key
            embedding_service.embed_document(document_id, upload_dir=self.upload_dir,
                                              db_path=self.db_path, provider=self.provider)
        return keys

    def search(self, query, top_k=5):
        with patch.dict(os.environ, self.offline):
            return service.search(query, db_path=self.db_path, provider=self.provider,
                                  top_k=top_k)

    def test_smoke_related_chunk_ranks_first_and_scores_follow_the_contract(self):
        keys = self.build_corpus(SMOKE_CORPUS)

        for query, expected_key in SMOKE_QUERIES.items():
            with self.subTest(query=query):
                with patch.object(self.provider, "embed_query",
                                  wraps=self.provider.embed_query) as spy:
                    started = time.perf_counter()
                    results = self.search(query, top_k=4).results
                    elapsed_ms = (time.perf_counter() - started) * 1000
                spy.assert_called_once_with(query)  # the provider adds "query: " itself
                print(f"\n  {query!r}: " + ", ".join(
                    f"{keys[r.chunk.chunk_id]}={r.score:.3f}" for r in results)
                    + f" ({elapsed_ms:.0f} ms)", file=sys.stderr)

                self.assertEqual(len(results), 4)
                self.assertEqual(keys[results[0].chunk.chunk_id], expected_key)
                query_vector = self.raw("query: " + query)
                for result in results:
                    passage = self.raw("passage: " + result.chunk.text)
                    self.assertAlmostEqual(result.score, dot(query_vector, passage), places=5)
                # With the wrong (passage) prefix for the question, the score
                # of the top chunk would be different: the prefix matters.
                wrong = dot(self.raw("passage: " + query),
                            self.raw("passage: " + results[0].chunk.text))
                self.assertGreater(abs(wrong - results[0].score), 1e-4)

    def test_smoke_same_query_gives_the_same_results(self):
        self.build_corpus(SMOKE_CORPUS)

        self.assertEqual(self.search("What is SQLite?"), self.search("What is SQLite?"))
        self.assertEqual(self.provider.embed_query("What is SQLite?"),
                         self.provider.embed_query("What is SQLite?"))

    def test_retrieval_baseline_recall(self):
        keys = self.build_corpus(PASSAGES)

        ranked = {}
        for query in QUERIES:
            ranked[query] = [keys[r.chunk.chunk_id] for r in self.search(query, top_k=5).results]
        recall = {k: recall_at_k(ranked, QUERIES, k) for k in (1, 3, 5)}

        def rank_of(query):
            ranks = [ranked[query].index(key) + 1 for key in QUERIES[query] if key in ranked[query]]
            return min(ranks) if ranks else "not in top 5"

        lines = [f"  {query!r}: expected {sorted(QUERIES[query])} at rank {rank_of(query)}; "
                 f"top 3 {ranked[query][:3]}" for query in QUERIES]
        print("\n  Retrieval baseline (development only, "
              f"{len(PASSAGES)} chunks, {len(QUERIES)} queries):\n" + "\n".join(lines)
              + "\n  " + "  ".join(f"Recall@{k}={value:.3f}" for k, value in recall.items()),
              file=sys.stderr)
        for k, floor in MIN_RECALL.items():
            self.assertGreaterEqual(recall[k], floor, f"Recall@{k} fell below the baseline floor")


if __name__ == "__main__":
    unittest.main()
