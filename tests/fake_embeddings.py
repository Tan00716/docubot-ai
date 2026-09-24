"""A tiny, deterministic stand-in for the real embedding model.

It lets storage, stale detection, batching and the API be tested quickly and
offline. The real model is tested separately in test_embedding_smoke.py.
"""

import hashlib
from pathlib import Path

from app.embeddings.config import VECTOR_DTYPE, EmbeddingConfig
from app.embeddings.models import EmbeddedText, EmbeddingContract
from app.embeddings.vectors import l2_normalize, to_float32

FAKE_MODEL_NAME = "fake/test-model"


def fake_vector(text: str, dimension: int) -> list[float]:
    """Numbers derived from the text's hash: same text -> same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i % len(digest)] - 127.5) / 127.5 for i in range(dimension)]


class FakeEmbeddingProvider:
    """Behaves like FastEmbedProvider, without any model.

    - texts longer than truncate_after characters are reported as truncated
    - batch_calls records the size of every embed_documents call
    """

    def __init__(self, dimension=8, model_name=FAKE_MODEL_NAME, embedding_version=1,
                 normalize=True, batch_size=16, truncate_after=1000):
        self.config = EmbeddingConfig(
            cache_dir=Path("unused-cache"),
            model_name=model_name,
            embedding_version=embedding_version,
            normalize_embeddings=normalize,
            batch_size=batch_size,
        )
        self.contract = EmbeddingContract(
            model_name=model_name,
            embedding_version=embedding_version,
            dimension=dimension,
            dtype=VECTOR_DTYPE,
            normalized=normalize,
        )
        self.truncate_after = truncate_after
        self.batch_calls: list[int] = []
        self.embedded_texts: list[str] = []

    def _vector(self, text):
        values = fake_vector(text, self.contract.dimension)
        return to_float32(l2_normalize(values) if self.contract.normalized else values)

    def embed_documents(self, texts):
        texts = list(texts)
        self.batch_calls.append(len(texts))
        self.embedded_texts.extend(texts)
        return [EmbeddedText(self._vector(text), len(text) > self.truncate_after)
                for text in texts]

    def embed_query(self, text):
        return self._vector(text)
