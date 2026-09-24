"""Unit tests for embeddings: config, contract, vectors, provider logic and planning.

No model is downloaded or loaded here: FastEmbed's TextEmbedding class and
the Hugging Face download are replaced by small fakes. The real model is
tested in test_embedding_smoke.py.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import dataclasses
import json
import logging
import math
import shutil
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.document_processing.models import Chunk
from app.embeddings import config as config_module
from app.embeddings import provider as provider_module
from app.embeddings.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    EMBEDDING_VERSION,
    MAX_BATCH_SIZE,
    MULTILINGUAL_E5_SMALL,
    SUPPORTED_MODELS,
    VECTOR_DTYPE,
    EmbeddingConfig,
    EmbeddingModelSpec,
    get_model_spec,
    validate_batch_size,
)
from app.embeddings.models import (
    EmbeddingContract,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingModelMismatchError,
    EmbeddingModelUnavailableError,
    EmbeddingRecord,
    InvalidEmbeddingConfigError,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.provider import FastEmbedProvider, build_contract, download_model_files
from app.embeddings.service import plan_embeddings
from app.embeddings.vectors import (
    deserialize_vector,
    l2_normalize,
    serialize_vector,
    text_sha256,
    to_float32,
)
from fake_embeddings import make_fake_contract

logging.getLogger("app").setLevel(logging.CRITICAL)  # failures are provoked on purpose

CACHE = Path("unused-cache")
FAKE_MODEL = "fake/model"
OLD_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# --- Configuration ------------------------------------------------------------


class EmbeddingConfigTests(unittest.TestCase):
    def test_defaults(self):
        config = EmbeddingConfig(cache_dir=CACHE)

        self.assertEqual(config.model_name, "intfloat/multilingual-e5-small")
        self.assertEqual(config.model_name, DEFAULT_MODEL_NAME)
        self.assertEqual(config.embedding_version, EMBEDDING_VERSION)
        self.assertTrue(config.normalize_embeddings)
        self.assertEqual(config.batch_size, DEFAULT_BATCH_SIZE)
        self.assertEqual(DEFAULT_BATCH_SIZE, 16)
        self.assertEqual(VECTOR_DTYPE, "float32")

    def test_embedding_version_was_bumped_for_the_new_model(self):
        # Version 1 was paraphrase-multilingual-MiniLM-L12-v2 (Batch 5).
        self.assertEqual(EMBEDDING_VERSION, 2)

    def test_e5_spec_matches_the_official_model_files(self):
        spec = get_model_spec(DEFAULT_MODEL_NAME)

        self.assertIs(spec, MULTILINGUAL_E5_SMALL)
        self.assertEqual((spec.dimension, spec.max_tokens, spec.pooling, spec.license),
                         (384, 512, "mean", "mit"))
        self.assertEqual((spec.passage_prefix, spec.query_prefix), ("passage: ", "query: "))
        self.assertEqual(spec.model_file, "onnx/model.onnx")
        self.assertEqual(spec.revision, "614241f622f53c4eeff9890bdc4f31cfecc418b3")
        self.assertRegex(spec.revision, r"^[0-9a-f]{40}$", "a full git commit, not a branch")

    def test_only_listed_models_are_supported(self):
        self.assertEqual(set(SUPPORTED_MODELS), {"intfloat/multilingual-e5-small"})
        for name in [OLD_MODEL, "intfloat/multilingual-e5-base", "someone/unknown", ""]:
            with self.subTest(model=name):
                with self.assertRaises(UnsupportedEmbeddingModelError):
                    get_model_spec(name)

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


# --- Contract -----------------------------------------------------------------


class EmbeddingContractTests(unittest.TestCase):
    def test_default_contract_is_the_e5_contract(self):
        contract = build_contract(EmbeddingConfig(cache_dir=CACHE))

        self.assertEqual(contract, EmbeddingContract(
            model_name="intfloat/multilingual-e5-small",
            model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3",
            embedding_version=2, dimension=384, max_tokens=512,
            passage_prefix="passage: ", query_prefix="query: ",
            dtype="float32", normalized=True))

    def test_passage_and_query_inputs(self):
        contract = build_contract(EmbeddingConfig(cache_dir=CACHE))

        self.assertEqual(contract.passage_input("FastAPI is a Python web framework."),
                         "passage: FastAPI is a Python web framework.")
        self.assertEqual(contract.query_input("What is FastAPI?"), "query: What is FastAPI?")
        self.assertEqual(contract.passage_input("你好"), "passage: 你好")

    def test_unsupported_model_has_no_contract(self):
        with self.assertRaises(UnsupportedEmbeddingModelError):
            build_contract(EmbeddingConfig(cache_dir=CACHE, model_name=OLD_MODEL))


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



# --- Provider (FastEmbed and the download replaced by fakes) ---------------------

FAKE_SPEC = EmbeddingModelSpec(
    name=FAKE_MODEL, revision="c" * 40, model_file="onnx/model.onnx", dimension=4,
    max_tokens=512, pooling="mean", passage_prefix="passage: ", query_prefix="query: ",
    license="mit",
)


class FakeArray:
    """Mimics a numpy array: FastEmbed returns these, the provider calls tolist()."""

    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


def make_fake_text_embedding(dimension=4, vector=None, fail_load=False, fail_embed=False,
                             token_limit=5, tokenizer_max_length=512, built_in=()):
    """A fake FastEmbed TextEmbedding class with its own counters.

    The fake tokenizer counts words; an input with more than token_limit
    words is reported as overflowing (truncated).
    """

    class FakeTextEmbedding:
        loads = 0
        instances = []
        custom_models = []

        def __init__(self, model_name, cache_dir, specific_model_path, providers):
            type(self).loads += 1
            if fail_load:
                raise RuntimeError("download failed at C:\\secret\\cache\\path")
            self.model_name, self.cache_dir = model_name, cache_dir
            self.specific_model_path, self.providers = specific_model_path, providers
            self.embed_calls = []
            tokenizer = SimpleNamespace(
                encode=lambda text: SimpleNamespace(
                    overflowing=["cut"] if len(text.split()) > token_limit else []),
                truncation={"max_length": tokenizer_max_length},
            )
            self.model = SimpleNamespace(tokenizer=tokenizer)
            type(self).instances.append(self)

        @staticmethod
        def list_supported_models():
            return [{"model": name, "dim": 2} for name in built_in]

        @classmethod
        def add_custom_model(cls, **description):
            cls.custom_models.append(description)

        def _vector(self):
            return FakeArray(vector if vector is not None else [3.0, 4.0] + [0.0] * (dimension - 2))

        def embed(self, texts, batch_size):
            if fail_embed:
                raise RuntimeError("onnx failure with internal detail")
            texts = list(texts)
            self.embed_calls.append((texts, batch_size))
            return (self._vector() for _ in texts)

    return FakeTextEmbedding


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        self.model_dir = Path(tempfile.mkdtemp(prefix="docubot-fake-model-"))
        self.addCleanup(shutil.rmtree, self.model_dir, ignore_errors=True)
        self.write_model_config(hidden_size=4)
        self.downloads = []
        for patcher in [
            patch.dict(config_module.SUPPORTED_MODELS, {FAKE_MODEL: FAKE_SPEC}),
            patch.object(provider_module, "_registered_models", {}),
            patch.object(provider_module, "download_model_files", side_effect=self.fake_download),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_model_config(self, **config):
        (self.model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")

    def fake_download(self, spec, cache_dir):
        self.downloads.append((spec.name, spec.revision, cache_dir))
        return self.model_dir

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
            model_name=FAKE_MODEL, model_revision="c" * 40, embedding_version=EMBEDDING_VERSION,
            dimension=4, max_tokens=512, passage_prefix="passage: ", query_prefix="query: ",
            dtype="float32", normalized=True))
        self.assertEqual((fake.loads, self.downloads), (0, []))

    def test_unsupported_model_is_rejected(self):
        with self.assertRaises(UnsupportedEmbeddingModelError):
            self.make_provider(model_name="someone/unknown-model")
        with self.assertRaises(UnsupportedEmbeddingModelError):
            self.make_provider(model_name=OLD_MODEL)

    def test_model_is_registered_with_fastembed_once_with_the_spec_settings(self):
        provider, fake = self.make_provider()

        provider.embed_documents(["a"])
        FastEmbedProvider(provider.config).embed_documents(["b"])

        self.assertEqual(fake.custom_models, [{
            "model": FAKE_MODEL, "pooling": provider_module.PoolingType.MEAN,
            "normalization": False, "sources": provider_module.ModelSource(hf=FAKE_MODEL),
            "dim": 4, "model_file": "onnx/model.onnx", "license": "mit",
        }])

    def test_changed_spec_for_an_already_registered_model_is_refused(self):
        provider, fake = self.make_provider()
        provider.embed_documents(["a"])
        bumped = dataclasses.replace(FAKE_SPEC, revision="d" * 40)

        with patch.dict(config_module.SUPPORTED_MODELS, {FAKE_MODEL: bumped}):
            with self.assertRaises(UnsupportedEmbeddingModelError):
                FastEmbedProvider(provider.config).embed_documents(["b"])

        self.assertEqual(len(fake.custom_models), 1, "FastEmbed keeps only the first spec")

    def test_model_that_fastembed_already_defines_is_refused(self):
        provider, _ = self.make_provider(built_in=[FAKE_MODEL.upper()])

        with self.assertRaises(UnsupportedEmbeddingModelError):
            provider.embed_documents(["a"])

    def test_pinned_files_are_loaded_on_the_cpu(self):
        provider, fake = self.make_provider()

        provider.embed_documents(["a"])

        self.assertEqual(self.downloads, [(FAKE_MODEL, "c" * 40, CACHE)])
        model = fake.instances[0]
        self.assertEqual(model.specific_model_path, str(self.model_dir))
        self.assertEqual(model.providers, ["CPUExecutionProvider"])

    def test_one_text_gives_one_normalized_float32_vector(self):
        provider, _ = self.make_provider()

        [result] = provider.embed_documents(["Hello"])

        self.assertEqual(len(result.vector), 4)
        self.assertEqual(result.vector, to_float32([0.6, 0.8, 0.0, 0.0]))
        self.assertEqual(result.vector, to_float32(result.vector), "values must be float32")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in result.vector)), 1.0, places=6)
        self.assertTrue(all(isinstance(x, float) for x in result.vector))

    def test_documents_are_embedded_as_passages_in_one_model_call(self):
        provider, fake = self.make_provider()
        texts = ["one", "two", "三"]

        results = provider.embed_documents(texts)

        self.assertEqual(len(results), 3)
        self.assertEqual(fake.instances[0].embed_calls,
                         [(["passage: one", "passage: two", "passage: 三"], 3)])
        self.assertEqual(texts, ["one", "two", "三"], "the caller's texts are not changed")

    def test_reported_input_hash_is_of_the_prefixed_text(self):
        provider, _ = self.make_provider()

        [result] = provider.embed_documents(["FastAPI is a Python web framework."])

        self.assertEqual(result.input_sha256,
                         text_sha256("passage: FastAPI is a Python web framework."))
        self.assertNotEqual(result.input_sha256,
                            text_sha256("FastAPI is a Python web framework."))

    def test_query_is_embedded_with_the_query_prefix_only(self):
        provider, fake = self.make_provider()

        vector = provider.embed_query("What is FastAPI?")

        self.assertEqual(vector, to_float32([0.6, 0.8, 0.0, 0.0]))
        self.assertEqual(fake.instances[0].embed_calls, [(["query: What is FastAPI?"], 1)])

    def test_passage_and_query_prefixes_are_never_swapped(self):
        provider, fake = self.make_provider()

        provider.embed_documents(["FastAPI is a Python web framework."])
        provider.embed_query("What is FastAPI?")

        [(document_inputs, _), (query_inputs, _)] = fake.instances[0].embed_calls
        self.assertTrue(all(i.startswith("passage: ") and "query: " not in i
                            for i in document_inputs))
        self.assertTrue(all(i.startswith("query: ") and "passage: " not in i
                            for i in query_inputs))

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

        self.assertEqual((fake.loads, len(self.downloads)), (1, 1))

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

    def test_truncation_is_measured_on_the_prefixed_input(self):
        provider, _ = self.make_provider(token_limit=4)

        # "passage: a b c" has 4 "tokens"; the prefix pushes "a b c d" over the limit.
        short, borderline = provider.embed_documents(["a b c", "a b c d"])

        self.assertFalse(short.truncated)
        self.assertTrue(borderline.truncated)

    def test_model_load_failure_gives_safe_error(self):
        provider, _ = self.make_provider(fail_load=True)

        with self.assertRaises(EmbeddingModelUnavailableError) as caught:
            provider.embed_documents(["text"])

        self.assertNotIn("secret", caught.exception.safe_message)
        self.assertIn("downloaded once", caught.exception.safe_message)

    def test_download_failure_gives_safe_error(self):
        provider, _ = self.make_provider()

        with patch.object(provider_module, "download_model_files",
                          side_effect=OSError("C:\\secret\\path")):
            with self.assertRaises(EmbeddingModelUnavailableError):
                provider.embed_documents(["text"])

    def test_model_files_with_another_dimension_are_refused(self):
        self.write_model_config(hidden_size=768)
        provider, _ = self.make_provider()

        with self.assertRaises(EmbeddingModelMismatchError):
            provider.embed_documents(["text"])

    def test_tokenizer_with_another_token_limit_is_refused(self):
        for limit in [128, 514, None]:
            with self.subTest(max_length=limit):
                provider, _ = self.make_provider(tokenizer_max_length=limit)

                with self.assertRaises(EmbeddingModelMismatchError):
                    provider.embed_documents(["text"])

    def test_missing_model_config_is_refused(self):
        (self.model_dir / "config.json").unlink()
        provider, _ = self.make_provider()

        with self.assertRaises(EmbeddingModelMismatchError) as caught:
            provider.embed_documents(["text"])

        self.assertNotIn(str(self.model_dir), caught.exception.safe_message)

    def test_model_failure_during_embedding_gives_safe_error(self):
        provider, _ = self.make_provider(fail_embed=True)

        with self.assertRaises(EmbeddingError) as caught:
            provider.embed_documents(["text"])

        self.assertNotIn("internal detail", caught.exception.safe_message)

    def test_wrong_vector_dimension_is_rejected(self):
        provider, _ = self.make_provider(vector=[1.0, 2.0, 3.0])  # contract says 4

        with self.assertRaises(EmbeddingDimensionError):
            provider.embed_documents(["text"])
        with self.assertRaises(EmbeddingDimensionError):
            provider.embed_query("text")

    def test_nan_or_zero_vectors_are_rejected(self):
        for vector in [[math.nan, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]:
            with self.subTest(vector=vector):
                provider, _ = self.make_provider(vector=vector)

                with self.assertRaises(EmbeddingError):
                    provider.embed_documents(["text"])


class DownloadTests(unittest.TestCase):
    """download_model_files with huggingface_hub.snapshot_download replaced."""

    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="docubot-snapshot-"))
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)
        self.calls = []

    def fake_snapshot_download(self, **options):
        self.calls.append(options)
        return str(self.folder)

    def download(self):
        with patch.object(provider_module, "snapshot_download",
                          side_effect=self.fake_snapshot_download):
            return download_model_files(MULTILINGUAL_E5_SMALL, CACHE)

    def test_pinned_revision_and_only_the_needed_files_are_requested(self):
        self.download()

        for call in self.calls:
            self.assertEqual(call["repo_id"], "intfloat/multilingual-e5-small")
            self.assertEqual(call["revision"], "614241f622f53c4eeff9890bdc4f31cfecc418b3")
            self.assertEqual(call["cache_dir"], str(CACHE))
            self.assertEqual(call["allow_patterns"], [
                "config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "onnx/model.onnx"])
            self.assertFalse(any(p.endswith((".bin", ".pt", ".pkl", ".py", ".safetensors"))
                                 for p in call["allow_patterns"]))

    def test_complete_local_copy_is_used_without_network(self):
        for name in provider_module.model_files(MULTILINGUAL_E5_SMALL):
            (self.folder / name).parent.mkdir(parents=True, exist_ok=True)
            (self.folder / name).write_bytes(b"x")

        self.assertEqual(self.download(), self.folder)
        self.assertEqual([call.get("local_files_only") for call in self.calls], [True])

    def test_incomplete_local_copy_is_downloaded_again(self):
        (self.folder / "config.json").write_bytes(b"{}")  # the ONNX file is missing

        self.download()

        self.assertEqual([call.get("local_files_only") for call in self.calls], [True, None])


# --- Planning (which chunks need a new vector) -----------------------------------


def make_chunk(index, text, document_id="a" * 32):
    return Chunk(
        chunk_id=f"{document_id}_v1_s1200_o200_{index:05d}", document_id=document_id,
        chunk_index=index, text=text, char_count=len(text), source_filename="a.txt",
        file_type=".txt", source_locations=({"block": index + 1},), chunking_version=1,
        chunk_size=1200, chunk_overlap=200,
    )


CONTRACT = make_fake_contract(model_name="fake/model", dimension=8)


def make_record(chunk, **changes):
    values = dict(chunk_id=chunk.chunk_id, model_name=CONTRACT.model_name,
                  model_revision=CONTRACT.model_revision,
                  embedding_version=CONTRACT.embedding_version, dimension=CONTRACT.dimension,
                  max_tokens=CONTRACT.max_tokens, passage_prefix=CONTRACT.passage_prefix,
                  dtype=CONTRACT.dtype, normalized=CONTRACT.normalized,
                  text_sha256=text_sha256(chunk.text),
                  input_sha256=text_sha256(CONTRACT.passage_input(chunk.text)),
                  truncated=False, created_at="t")
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
            "input without prefix": {"input_sha256": text_sha256("text")},
            "input with query prefix": {"input_sha256": text_sha256("query: text")},
            "other model": {"model_name": "other/model"},
            "old Batch 5 model": {"model_name": OLD_MODEL, "embedding_version": 1,
                                  "model_revision": "", "max_tokens": 0,
                                  "passage_prefix": "", "input_sha256": ""},
            "other revision": {"model_revision": "d" * 40},
            "other version": {"embedding_version": 1},
            "other dimension": {"dimension": 16},
            "other token limit": {"max_tokens": 128},
            "other passage prefix": {"passage_prefix": ""},
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

    def test_changing_only_the_query_prefix_keeps_stored_passages_valid(self):
        # Stored vectors are passages; the query prefix never touched them.
        chunk = make_chunk(0, "text")
        other_query = make_fake_contract(model_name="fake/model", dimension=8,
                                         query_prefix="question: ")

        plan = plan_embeddings([chunk], {chunk.chunk_id: make_record(chunk)}, other_query)

        self.assertEqual(plan.valid, (chunk,))


if __name__ == "__main__":
    unittest.main()
