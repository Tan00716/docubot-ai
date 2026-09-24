"""A tiny, deterministic stand-in for the real embedding model.

It lets storage, stale detection, batching and the API be tested quickly and
offline. The real model is tested separately in test_embedding_smoke.py.
It follows the same E5-style input contract as the real provider: documents
are embedded as "passage: <text>" and questions as "query: <text>".
"""

import hashlib
from pathlib import Path

from app.embeddings.config import VECTOR_DTYPE, EmbeddingConfig
from app.embeddings.models import EmbeddedText, EmbeddingContract
from app.embeddings.vectors import l2_normalize, text_sha256, to_float32

FAKE_MODEL_NAME = "fake/test-model"
FAKE_REVISION = "f" * 40


def fake_vector(text: str, dimension: int) -> list[float]:
    """Numbers derived from the text's hash: same text -> same vector."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [(digest[i % len(digest)] - 127.5) / 127.5 for i in range(dimension)]


def make_fake_contract(dimension=8, model_name=FAKE_MODEL_NAME, embedding_version=2,
                       normalize=True, revision=FAKE_REVISION, max_tokens=512,
                       passage_prefix="passage: ", query_prefix="query: "):
    return EmbeddingContract(
        model_name=model_name,
        model_revision=revision,
        embedding_version=embedding_version,
        dimension=dimension,
        max_tokens=max_tokens,
        passage_prefix=passage_prefix,
        query_prefix=query_prefix,
        dtype=VECTOR_DTYPE,
        normalized=normalize,
    )


class FakeEmbeddingProvider:
    """Behaves like FastEmbedProvider, without any model.

    - texts longer than truncate_after characters are reported as truncated
    - batch_calls records the size of every embed_documents call
    - model_inputs records exactly what the "model" was given (prefix included)
    - wrong_passage_prefix simulates a provider bug: it embeds with another
      prefix than the contract says, and reports that input honestly
    """

    def __init__(self, dimension=8, model_name=FAKE_MODEL_NAME, embedding_version=2,
                 normalize=True, batch_size=16, truncate_after=1000,
                 revision=FAKE_REVISION, max_tokens=512, wrong_passage_prefix=None):
        self.config = EmbeddingConfig(
            cache_dir=Path("unused-cache"),
            model_name=model_name,
            embedding_version=embedding_version,
            normalize_embeddings=normalize,
            batch_size=batch_size,
        )
        self.contract = make_fake_contract(
            dimension=dimension, model_name=model_name, embedding_version=embedding_version,
            normalize=normalize, revision=revision, max_tokens=max_tokens,
        )
        self.truncate_after = truncate_after
        self.wrong_passage_prefix = wrong_passage_prefix
        self.batch_calls: list[int] = []
        self.embedded_texts: list[str] = []  # the original chunk texts
        self.model_inputs: list[str] = []

    def _vector(self, model_input):
        values = fake_vector(model_input, self.contract.dimension)
        return to_float32(l2_normalize(values) if self.contract.normalized else values)

    def _passage_input(self, text):
        if self.wrong_passage_prefix is not None:
            return self.wrong_passage_prefix + text
        return self.contract.passage_input(text)

    def passage_vector(self, text):
        """The vector a correct provider stores for this chunk text."""
        return self._vector(self.contract.passage_input(text))

    def embed_documents(self, texts):
        texts = list(texts)
        self.batch_calls.append(len(texts))
        self.embedded_texts.extend(texts)
        inputs = [self._passage_input(text) for text in texts]
        self.model_inputs.extend(inputs)
        return [EmbeddedText(self._vector(model_input), len(text) > self.truncate_after,
                             text_sha256(model_input))
                for text, model_input in zip(texts, inputs)]

    def embed_query(self, text):
        model_input = self.contract.query_input(text)
        self.model_inputs.append(model_input)
        return self._vector(model_input)
