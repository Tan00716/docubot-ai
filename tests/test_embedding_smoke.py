"""Smoke test with the REAL local model (opt-in, not part of the normal run).

It loads sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 from
storage/model_cache/ (downloading it once, ~240 MB, if it is not there yet),
embeds short English and Chinese texts on the CPU, and stores and reloads the
vectors in a temporary database. It does NOT judge retrieval quality.

Run it explicitly from the project root (PowerShell):

    $env:DOCUBOT_RUN_MODEL_SMOKE_TEST = "1"
    .venv\\Scripts\\python.exe -m unittest tests.test_embedding_smoke -v
"""

import math
import os
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for "python -m unittest tests.x"

from app import api  # noqa: E402
from app.embeddings import service, storage  # noqa: E402
from app.embeddings.config import DEFAULT_MODEL_NAME, EmbeddingConfig  # noqa: E402
from app.embeddings.provider import FastEmbedProvider  # noqa: E402
from test_embedding_storage import EmbeddingTestCase  # noqa: E402

TEXTS = [
    "FastAPI is a Python web framework.",
    "FastAPI 是一个用于构建 API 的 Python Web Framework。",
    "FastAPI is used to build web APIs with Python.",
]


def model_is_cached(cache_dir: Path) -> bool:
    return any(cache_dir.glob("models--qdrant--paraphrase-multilingual-MiniLM-L12-v2*"))


@unittest.skipUnless(os.environ.get("DOCUBOT_RUN_MODEL_SMOKE_TEST") == "1",
                     "set DOCUBOT_RUN_MODEL_SMOKE_TEST=1 to run the real model smoke test")
class RealModelSmokeTest(EmbeddingTestCase):
    @classmethod
    def setUpClass(cls):
        cache_dir = api.MODEL_CACHE_DIR
        # If the model is already cached, forbid network access completely,
        # proving that loading and embedding run locally.
        offline = {"HF_HUB_OFFLINE": "1"} if model_is_cached(cache_dir) else {}
        started = time.perf_counter()
        with patch.dict(os.environ, offline):
            cls.provider = FastEmbedProvider(EmbeddingConfig(cache_dir=cache_dir))
            cls.results = cls.provider.embed_documents(TEXTS)
        cls.load_seconds = time.perf_counter() - started
        cls.offline = bool(offline)

    def test_model_loads_locally_and_gives_384_dimensional_unit_vectors(self):
        print(f"\n  model load + first batch: {self.load_seconds:.1f}s "
              f"(offline={self.offline})", file=sys.stderr)
        self.assertEqual(self.provider.contract.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(self.provider.contract.dimension, 384)
        for result in self.results:
            self.assertEqual(len(result.vector), 384)
            self.assertAlmostEqual(math.sqrt(sum(x * x for x in result.vector)), 1.0, places=5)
            self.assertFalse(result.truncated)

    def test_same_text_gives_the_same_vector_and_query_uses_the_same_space(self):
        again = self.provider.embed_documents(TEXTS[:1])[0].vector

        self.assertEqual(again, self.results[0].vector)
        self.assertEqual(self.provider.embed_query(TEXTS[0]), self.results[0].vector)

    def test_long_text_is_flagged_as_truncated(self):
        [result] = self.provider.embed_documents([" ".join(f"word{i}" for i in range(300))])

        self.assertTrue(result.truncated)

    def test_real_vectors_are_stored_and_loaded(self):
        document_id = self.chunked_document(
            "\n\n".join(TEXTS).encode("utf-8"), config=None)

        started = time.perf_counter()
        summary = service.embed_document(document_id, upload_dir=self.upload_dir,
                                         db_path=self.db_path, provider=self.provider)
        print(f"\n  embedded {summary.embedded_count} chunk(s) in "
              f"{time.perf_counter() - started:.2f}s", file=sys.stderr)

        chunk = self.chunks(document_id)[0]
        stored = storage.get_embedding(self.db_path, chunk.chunk_id)
        self.assertEqual(stored.record.dimension, 384)
        self.assertEqual(stored.vector, self.provider.embed_query(chunk.text))


if __name__ == "__main__":
    unittest.main()
