"""Unit tests for embeddings: config, vectors, provider logic and planning.

No model is downloaded or loaded here: FastEmbed's TextEmbedding class is
replaced by a small fake. The real model is tested in test_embedding_smoke.py.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import logging
import math
import struct
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.document_processing.models import Chunk
from app.embeddings import provider as provider_module
from app.embeddings.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    EMBEDDING_VERSION,
    MAX_BATCH_SIZE,
    VECTOR_DTYPE,
    EmbeddingConfig,
    validate_batch_size,
)
from app.embeddings.models import (
    EmbeddingContract,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingModelUnavailableError,
    EmbeddingRecord,
    InvalidEmbeddingConfigError,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.provider import FastEmbedProvider, model_dimension
from app.embeddings.service import plan_embeddings
from app.embeddings.vectors import (
    deserialize_vector,
    l2_normalize,
    serialize_vector,
    text_sha256,
    to_float32,
)

logging.getLogger("app").setLevel(logging.CRITICAL)  # failures are provoked on purpose

CACHE = Path("unused-cache")
FAKE_MODEL = "fake/model"


# --- Configuration ------------------------------------------------------------


class EmbeddingConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = EmbeddingConfig(cache_dir=CACHE)

        self.assertEqual(config.model_name,
                         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.embedding_version, EMBEDDING_VERSION)
        self.assertTrue(config.normalize_embeddings)
        self.assertEqual(config.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(DEFAULT_BATCH_SIZE, 16)
        self.assertEqual(VECTOR_DTYPE, "float32")

    def test_model_name_and_version_are_configurable(self):
        config = EmbeddingConfig(cache_dir=CACHE, model_name="other/model", embedding_version=7)

        self.assertEqual((config.model_name, config.embedding_version), ("other/model", 7))

    def test_invalid_settings_are_rejected(self):
        cases = [
            {"model_name": ""},
            {"model_name": "   "},
            {"model_name": None},
            {"embedding_version": 0},
            {"embedding_version": -1},
            {"embedding_version": True},
            {"embedding_version": "1"},
            {"normalize_embeddings": "yes"},
            {"batch_size": 0},
            {"batch_size": MAX_BATCH_SIZE + 1},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(InvalidEmbeddingConfigError):
                    EmbeddingConfig(cache_dir=CACHE, **changes)

    def test_cache_dir_must_be_a_path(self):
        with self.assertRaises(InvalidEmbeddingConfigError):
            EmbeddingConfig(cache_dir="storage/model_cache")

    def test_batch_size_validation(self):
        for good in [1, 16, MAX_BATCH_SIZE]:
            self.assertEqual(validate_batch_size(good), good)
        for bad in [0, -3, MAX_BATCH_SIZE + 1, 2.5, "8", True, None]:
            with self.subTest(batch_size=bad):
                with self.assertRaises(InvalidEmbeddingConfigError) as caught:
                    validate_batch_size(bad)

                self.assertEqual(caught.exception.safe_message,
                                 f"batch_size must be a whole number from 1 to {MAX_BATCH_SIZE}.")


# --- Vectors ------------------------------------------------------------------


class TextHashTests(unittest.TestCase):
    def test_hash_is_deterministic_and_matches_sha256(self):
        self.assertEqual(
            text_sha256("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        self.assertEqual(text_sha256("你好 DocuBot"), text_sha256("你好 DocuBot"))

    def test_any_text_change_changes_the_hash(self):
        original = text_sha256("FastAPI is a web framework.")

        for changed in ["FastAPI is a web framework!", "FastAPI is a web framework. ",
                        "fastAPI is a web framework.", "FastAPI  is a web framework."]:
            with self.subTest(changed=changed):
                self.assertNotEqual(text_sha256(changed), original)


class VectorSerializationTests(unittest.TestCase):
    def test_vector_is_packed_as_little_endian_float32(self):
        blob = serialize_vector([1.0, -2.0], dimension=2)

        self.assertEqual(blob, b"\x00\x00\x80\x3f" + b"\x00\x00\x00\xc0")
        self.assertEqual(len(serialize_vector([0.5] * 384, dimension=384)), 384 * 4)

    def test_round_trip_is_exact_for_float32_values(self):
        vector = to_float32([0.1, -0.25, 3.14159, 1e-8])

        self.assertEqual(deserialize_vector(serialize_vector(vector, 4), 4), vector)

    def test_round_trip_of_any_floats_is_within_float32_precision(self):
        values = [0.1, 1 / 3, -2 / 7, 123.456]

        restored = deserialize_vector(serialize_vector(values, 4), 4)

        for original, back in zip(values, restored):
            self.assertTrue(math.isclose(original, back, rel_tol=1e-6))

    def test_serialization_is_deterministic(self):
        vector = [0.3, 0.4, 0.5]

        self.assertEqual(serialize_vector(vector, 3), serialize_vector(list(vector), 3))

    def test_wrong_dimension_is_rejected(self):
        with self.assertRaises(ValueError):
            serialize_vector([1.0, 2.0], dimension=3)

    def test_nan_and_infinity_are_rejected(self):
        for bad in [math.nan, math.inf, -math.inf]:
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    serialize_vector([1.0, bad], dimension=2)

    def test_malformed_blobs_are_rejected(self):
        cases = [b"", b"\x00" * 7, b"\x00" * 12, "not bytes", None,
                 struct.pack("<2f", 1.0, math.nan)]
        for blob in cases:
            with self.subTest(blob=blob):
                with self.assertRaises(ValueError):
                    deserialize_vector(blob, dimension=2)

    def test_l2_normalize_gives_length_one(self):
        vector = l2_normalize([3.0, 4.0])

        self.assertEqual(vector, (0.6, 0.8))
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in l2_normalize([1.5, -2, 7]))), 1.0)

    def test_zero_vector_cannot_be_normalized(self):
        with self.assertRaises(ValueError):
            l2_normalize([0.0, 0.0])


# --- Provider (FastEmbed replaced by a fake) ------------------------------------


class FakeArray:
    """Mimics a numpy array: FastEmbed returns these, the provider calls tolist()."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def make_fake_text_embedding(dimension=4, vector=None, fail_load=False, fail_embed=False,
                             token_limit=5):
    """A fake FastEmbed TextEmbedding class with its own counters."""

    class FakeTextEmbedding:
        loads = 0
        instances = []

        def __init__(self, model_name, cache_dir):
            type(self).loads += 1
            if fail_load:
                raise RuntimeError("download failed at C:\\secret\\cache\\path")
            self.model_name, self.cache_dir = model_name, cache_dir
            self.embed_calls = []
            tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(
                overflowing=["cut"] if len(text.split()) > token_limit else []))
            self.model = SimpleNamespace(tokenizer=tokenizer)
            type(self).instances.append(self)

        @staticmethod
        def list_supported_models():
            return [{"model": FAKE_MODEL, "dim": dimension}, {"model": "fake/other", "dim": 2}]

        def _vector(self):
            return FakeArray(vector if vector is not None else [3.0, 4.0] + [0.0] * (dimension - 2))

        def embed(self, texts, batch_size):
            if fail_embed:
                raise RuntimeError("onnx failure with internal detail")
            texts = list(texts)
            self.embed_calls.append((texts, batch_size))
            return (self._vector() for _ in texts)

        def query_embed(self, query):
            self.embed_calls.append(([query], "query"))
            return iter([self._vector()])

    return FakeTextEmbedding


class ProviderTestCase(unittest.TestCase):
    def make_provider(self, normalize=True, model_name=FAKE_MODEL, **fake_options):
        fake = make_fake_text_embedding(**fake_options)
        text_embedding_patch = patch.object(provider_module, "TextEmbedding", fake)
        text_embedding_patch.start()
        self.addCleanup(text_embedding_patch.stop)
        config = EmbeddingConfig(cache_dir=CACHE, model_name=model_name,
                                 normalize_embeddings=normalize)
        return FastEmbedProvider(config), fake


class ProviderTests(ProviderTestCase):
    def test_provider_reports_contract_without_loading_the_model(self):
        provider, fake = self.make_provider()

        self.assertEqual(provider.contract, EmbeddingContract(
            model_name=FAKE_MODEL, embedding_version=EMBEDDING_VERSION, dimension=4,
            dtype="float32", normalized=True))
        self.assertEqual(fake.loads, 0)

    def test_real_fastembed_list_gives_384_dimensions_for_default_model(self):
        # Reads FastEmbed's built-in model list only; nothing is downloaded.
        self.assertEqual(model_dimension(DEFAULT_MODEL_NAME), 384)

    def test_unsupported_model_is_rejected(self):
        with self.assertRaises(UnsupportedEmbeddingModelError):
            model_dimension("someone/unknown-model")
        with self.assertRaises(UnsupportedEmbeddingModelError):
            self.make_provider(model_name="someone/unknown-model")

    def test_one_text_gives_one_normalized_float32_vector(self):
        provider, _ = self.make_provider()

        [result] = provider.embed_documents(["Hello"])

        self.assertEqual(len(result.vector), 4)
        self.assertEqual(result.vector, to_float32([0.6, 0.8, 0.0, 0.0]))
        self.assertEqual(result.vector, to_float32(result.vector), "values must be float32")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in result.vector)), 1.0, places=6)
        self.assertTrue(all(isinstance(x, float) for x in result.vector))

    def test_several_texts_are_embedded_in_one_model_call(self):
        provider, fake = self.make_provider()

        results = provider.embed_documents(["one", "two", "three"])

        self.assertEqual(len(results), 3)
        self.assertEqual(fake.instances[0].embed_calls, [(["one", "two", "three"], 3)])

    def test_normalization_can_be_switched_off(self):
        provider, _ = self.make_provider(normalize=False)

        [result] = provider.embed_documents(["Hello"])

        self.assertEqual(result.vector, (3.0, 4.0, 0.0, 0.0))
        self.assertFalse(provider.contract.normalized)

    def test_model_is_loaded_once_and_reused(self):
        provider, fake = self.make_provider()

        provider.embed_documents(["a"])
        provider.embed_documents(["b", "c"])
        provider.embed_query("d")

        self.assertEqual(fake.loads, 1)

    def test_concurrent_first_calls_load_the_model_once(self):
        provider, fake = self.make_provider()
        threads = [threading.Thread(target=provider.embed_documents, args=(["x"],))
                   for _ in range(8)]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(fake.loads, 1)

    def test_empty_list_needs_no_model(self):
        provider, fake = self.make_provider()

        self.assertEqual(provider.embed_documents([]), [])
        self.assertEqual(fake.loads, 0)

    def test_empty_text_is_rejected(self):
        provider, _ = self.make_provider()

        for texts in [[""], ["ok", "   "], [None]]:
            with self.subTest(texts=texts):
                with self.assertRaises(EmbeddingError):
                    provider.embed_documents(texts)
        with self.assertRaises(EmbeddingError):
            provider.embed_query("")

    def test_duplicate_texts_get_identical_vectors(self):
        provider, _ = self.make_provider()

        first, second = provider.embed_documents(["same", "same"])

        self.assertEqual(first, second)

    def test_truncated_texts_are_flagged(self):
        provider, _ = self.make_provider(token_limit=3)

        short, long = provider.embed_documents(["a b c", "a b c d e"])

        self.assertFalse(short.truncated)
        self.assertTrue(long.truncated)

    def test_model_load_failure_gives_safe_error(self):
        provider, _ = self.make_provider(fail_load=True)

        with self.assertRaises(EmbeddingModelUnavailableError) as caught:
            provider.embed_documents(["text"])

        self.assertNotIn("secret", caught.exception.safe_message)
        self.assertIn("downloaded once", caught.exception.safe_message)

    def test_model_failure_during_embedding_gives_safe_error(self):
        provider, _ = self.make_provider(fail_embed=True)

        with self.assertRaises(EmbeddingError) as caught:
            provider.embed_documents(["text"])

        self.assertNotIn("internal detail", caught.exception.safe_message)

    def test_wrong_vector_dimension_is_rejected(self):
        provider, _ = self.make_provider(vector=[1.0, 2.0, 3.0])  # contract says 4

        with self.assertRaises(EmbeddingDimensionError):
            provider.embed_documents(["text"])

    def test_nan_or_zero_vectors_are_rejected(self):
        for vector in [[math.nan, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]:
            with self.subTest(vector=vector):
                provider, _ = self.make_provider(vector=vector)

                with self.assertRaises(EmbeddingError):
                    provider.embed_documents(["text"])

    def test_query_uses_the_same_model_and_normalization(self):
        provider, fake = self.make_provider()

        vector = provider.embed_query("What is FastAPI?")

        self.assertEqual(vector, to_float32([0.6, 0.8, 0.0, 0.0]))
        self.assertEqual(fake.instances[0].embed_calls, [(["What is FastAPI?"], "query")])


# --- Planning (which chunks need a new vector) -----------------------------------


def make_chunk(index, text, document_id="a" * 32):
    return Chunk(
        chunk_id=f"{document_id}_v1_s1200_o200_{index:05d}", document_id=document_id,
        chunk_index=index, text=text, char_count=len(text), source_filename="a.txt",
        file_type=".txt", source_locations=({"block": index + 1},), chunking_version=1,
        chunk_size=1200, chunk_overlap=200,
    )


CONTRACT = EmbeddingContract(model_name="fake/model", embedding_version=1, dimension=8,
                             dtype="float32", normalized=True)


def make_record(chunk, **changes):
    values = dict(chunk_id=chunk.chunk_id, model_name=CONTRACT.model_name,
                  embedding_version=CONTRACT.embedding_version, dimension=CONTRACT.dimension,
                  dtype=CONTRACT.dtype, normalized=CONTRACT.normalized,
                  text_sha256=text_sha256(chunk.text), truncated=False, created_at="t")
    return EmbeddingRecord(**(values | changes))


class PlanEmbeddingsTests(unittest.TestCase):
    def test_missing_valid_and_stale_are_told_apart(self):
        chunks = [make_chunk(0, "zero"), make_chunk(1, "one"), make_chunk(2, "two")]
        records = {chunks[0].chunk_id: make_record(chunks[0]),
                   chunks[2].chunk_id: make_record(chunks[2], text_sha256="0" * 64)}

        plan = plan_embeddings(chunks, records, CONTRACT)

        self.assertEqual(plan.valid, (chunks[0],))
        self.assertEqual(plan.missing, (chunks[1],))
        self.assertEqual(plan.stale, (chunks[2],))
        self.assertEqual(plan.to_embed, [chunks[1], chunks[2]])

    def test_every_contract_field_makes_an_embedding_stale(self):
        chunk = make_chunk(0, "text")
        changes = {
            "text changed": {"text_sha256": text_sha256("old text")},
            "other model": {"model_name": "other/model"},
            "other version": {"embedding_version": 2},
            "other dimension": {"dimension": 16},
            "other normalization": {"normalized": False},
            "unsupported dtype": {"dtype": "float16"},
        }
        for reason, change in changes.items():
            with self.subTest(reason=reason):
                plan = plan_embeddings([chunk], {chunk.chunk_id: make_record(chunk, **change)},
                                       CONTRACT)

                self.assertEqual(plan.stale, (chunk,))
                self.assertEqual(plan.valid, ())

    def test_matching_embedding_is_valid(self):
        chunk = make_chunk(0, "text")

        plan = plan_embeddings([chunk], {chunk.chunk_id: make_record(chunk)}, CONTRACT)

        self.assertEqual(plan.valid, (chunk,))
        self.assertEqual(plan.to_embed, [])


if __name__ == "__main__":
    unittest.main()
