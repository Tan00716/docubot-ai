"""Tests for the Batch 6B embedding model comparison (no real model, no download).

Covers the candidate allowlist and contracts, model-specific prefixes,
isolation from the production model, offline behaviour, the frozen dataset,
the comparison metrics and the report. FastEmbed and the Hugging Face
download are replaced by fakes; the fake provider still runs the real
upload -> process -> chunk -> embed -> search pipeline.

Run from the project root:

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import dataclasses
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.document_processing.chunking import ChunkingConfig
from app.embeddings import provider as provider_module
from app.embeddings.config import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_VERSION,
    MULTILINGUAL_E5_SMALL,
    SUPPORTED_MODELS,
    EmbeddingConfig,
)
from app.embeddings.models import (
    EmbeddingContract,
    EmbeddingModelMismatchError,
    EmbeddingRecord,
    UnsupportedEmbeddingModelError,
)
from app.embeddings.provider import FastEmbedProvider, build_contract, model_files
from app.embeddings.vectors import text_sha256
from app.evaluation import compare_models, comparison, comparison_report, runner
from app.evaluation.candidates import (
    BGE_M3,
    CANDIDATES,
    E5_BASE,
    E5_SMALL,
    UnknownCandidateError,
    get_candidate,
    make_provider,
    missing_files,
    snapshot_folder,
    verify_official_files,
)
from app.evaluation.comparison import (
    CROSS_LANGUAGE_DIRECTIONS,
    DATASET_SHA256,
    EXPECTED_DIRECTION_COUNTS,
    IncompleteResultError,
    ModelResult,
    collect_result,
    dataset_fingerprint,
    validate_result,
)
from app.evaluation.corpus import PASSAGES
from app.evaluation.golden import DIRECTIONS, QUERIES, Evidence
from app.evaluation.metrics import QueryOutcome, summarize
from app.evaluation.resources import SystemMemory
from app.search import storage as search_storage
from fake_embeddings import FakeEmbeddingProvider
from test_embeddings import make_fake_text_embedding

logging.getLogger("app").setLevel(logging.CRITICAL)

CACHE = Path("unused-cache")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GB = 1024**3


def resources(**changes):
    values = {"download_bytes": 1, "weights_bytes": 1, "cache_bytes": 1, "load_seconds": 1.0,
              "embed_seconds": 1.0, "search_ms_mean": 1.0, "search_ms_max": 2.0,
              "private_bytes_before_load": 1, "private_bytes_after_load": 1,
              "peak_working_set": 1, "peak_private_bytes": 1, "embedding_batch_size": 16}
    return values | changes


def with_ranks(result: ModelResult, rank_of: dict[str, int | None]) -> ModelResult:
    """A copy of result where the listed queries have one item at the given rank."""
    queries = tuple(dataclasses.replace(q, ranks=(rank_of[q.query_id],))
                    if q.query_id in rank_of else q for q in result.queries)
    return dataclasses.replace(result, queries=queries)


class FakeRun:
    """One real-pipeline run with the fake provider, shared by the tests (built once)."""

    result: ModelResult
    kb: runner.KnowledgeBase
    provider: FakeEmbeddingProvider

    @classmethod
    def build(cls):
        if hasattr(cls, "result"):
            return
        root = Path(tempfile.mkdtemp(prefix="docubot-compare-test-"))
        cls.provider = FakeEmbeddingProvider()
        cls.kb = runner.build_knowledge_base(root / "kb", cls.provider, ChunkingConfig())
        results = runner.evaluate(cls.kb, cls.provider)
        cls.result = collect_result("e5-small", E5_SMALL.spec, cls.kb, results,
                                    checks={"dimension": True}, resources=resources())
        cls.root = root


def tearDownModule():
    if hasattr(FakeRun, "root"):
        shutil.rmtree(FakeRun.root, ignore_errors=True)


# --- Candidates, contracts and prefixes -------------------------------------------


class CandidateSpecTests(unittest.TestCase):
    def test_candidate_specs_match_the_official_files_checked_on_2026_09_28(self):
        # config.json hidden_size, tokenizer_config.json model_max_length,
        # 1_Pooling/config.json and the model card FAQ at the pinned revisions.
        expected = {
            "e5-base": ("intfloat/multilingual-e5-base", "d128750597153bb5987e10b1c3493a34e5a4502a",
                        768, 512, "mean", "query: ", "passage: ", ()),
            "bge-m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181",
                       1024, 8192, "cls", "", "",
                       ("onnx/model.onnx_data", "onnx/Constant_7_attr__value")),
        }
        for key, values in expected.items():
            spec = get_candidate(key).spec
            with self.subTest(key):
                self.assertEqual((spec.name, spec.revision, spec.dimension, spec.max_tokens,
                                  spec.pooling, spec.query_prefix, spec.passage_prefix,
                                  spec.external_data_files), values)
                self.assertEqual(spec.model_file, "onnx/model.onnx")

    def test_every_candidate_is_pinned_to_an_exact_commit(self):
        for candidate in CANDIDATES.values():
            with self.subTest(candidate.key):
                revision = candidate.spec.revision
                self.assertRegex(revision, r"^[0-9a-f]{40}$")
                self.assertNotIn(revision, {"main", "latest", "default"})

    def test_only_allowlisted_names_are_accepted(self):
        self.assertEqual(list(CANDIDATES), ["e5-small", "e5-base", "bge-m3"])
        for name in ("intfloat/multilingual-e5-large", "https://huggingface.co/BAAI/bge-m3",
                     "../e5-base", "main", ""):
            with self.subTest(name), self.assertRaises(UnknownCandidateError):
                get_candidate(name)

    def test_e5_models_require_query_and_passage_prefixes(self):
        for candidate in (E5_SMALL, E5_BASE):
            contract = make_provider(candidate, CACHE).contract
            with self.subTest(candidate.key):
                self.assertEqual(contract.query_input("什么是 WAL?"), "query: 什么是 WAL?")
                self.assertEqual(contract.passage_input("SQLite text"), "passage: SQLite text")

    def test_bge_m3_gets_the_raw_text_without_any_e5_prefix(self):
        contract = make_provider(BGE_M3, CACHE).contract

        self.assertEqual(contract.query_input("什么是 WAL?"), "什么是 WAL?")
        self.assertEqual(contract.passage_input("SQLite text"), "SQLite text")

    def test_only_tokenizer_config_and_onnx_files_are_downloaded(self):
        self.assertEqual(model_files(BGE_M3.spec), [
            "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "onnx/model.onnx", "onnx/model.onnx_data", "onnx/Constant_7_attr__value",
            "1_Pooling/config.json", "modules.json"])
        for candidate in CANDIDATES.values():
            with self.subTest(candidate.key):
                self.assertFalse(any(name.endswith((".py", ".bin", ".pt", ".pkl", ".safetensors"))
                                     for name in model_files(candidate.spec)))


def write_official_files(folder: Path, hidden_size=1024, max_length=8192, cls=True,
                         normalize=True):
    """Model files shaped like BGE-M3's official ones (values from the Hub, 2026-09-28)."""
    (folder / "1_Pooling").mkdir(parents=True, exist_ok=True)
    files = {
        "config.json": {"hidden_size": hidden_size},
        "tokenizer_config.json": {"model_max_length": max_length},
        "1_Pooling/config.json": {"word_embedding_dimension": hidden_size,
                                  "pooling_mode_cls_token": cls,
                                  "pooling_mode_mean_tokens": not cls},
        "modules.json": [{"type": "sentence_transformers.models.Transformer"},
                         {"type": "sentence_transformers.models.Pooling"}]
        + ([{"type": "sentence_transformers.models.Normalize"}] if normalize else []),
    }
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content), encoding="utf-8")


class OfficialFilesTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix="docubot-official-"))
        self.addCleanup(shutil.rmtree, self.folder, ignore_errors=True)

    def test_matching_files_are_accepted(self):
        write_official_files(self.folder)

        self.assertEqual(verify_official_files(BGE_M3, self.folder), [])

    def test_each_mismatch_is_reported(self):
        cases = {"dimension": {"hidden_size": 768}, "token limit": {"max_length": 512},
                 "pooling": {"cls": False}, "normalization": {"normalize": False}}
        for name, change in cases.items():
            with self.subTest(name):
                write_official_files(self.folder, **change)
                self.assertTrue(verify_official_files(BGE_M3, self.folder))


class CandidateProviderTests(unittest.TestCase):
    """The unchanged FastEmbedProvider with a candidate spec, FastEmbed faked."""

    def setUp(self):
        self.model_dir = Path(tempfile.mkdtemp(prefix="docubot-fake-candidate-"))
        self.addCleanup(shutil.rmtree, self.model_dir, ignore_errors=True)
        for patcher in [
            patch.object(provider_module, "_registered_models", {}),
            patch.object(provider_module, "download_model_files",
                         side_effect=lambda spec, cache_dir: self.model_dir),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def make(self, candidate, hidden_size=None, tokenizer_max_length=None):
        spec = candidate.spec
        (self.model_dir / "config.json").write_text(
            json.dumps({"hidden_size": hidden_size or spec.dimension}), encoding="utf-8")
        fake = make_fake_text_embedding(
            dimension=spec.dimension, token_limit=10_000,
            tokenizer_max_length=tokenizer_max_length or spec.max_tokens)
        text_embedding_patch = patch.object(provider_module, "TextEmbedding", fake)
        text_embedding_patch.start()
        self.addCleanup(text_embedding_patch.stop)
        return make_provider(candidate, CACHE), fake

    def test_model_receives_the_model_specific_inputs(self):
        expected = {E5_BASE: ("passage: 文本", "query: 问题"), BGE_M3: ("文本", "问题")}
        for candidate, (passage, query) in expected.items():
            with self.subTest(candidate.key):
                provider, fake = self.make(candidate)
                provider.embed_documents(["文本"])
                provider.embed_query("问题")
                calls = [texts for texts, _ in fake.instances[0].embed_calls]
                self.assertEqual(calls, [[passage], [query]])

    def test_bge_m3_is_registered_with_cls_pooling_and_normalized_by_docubot(self):
        provider, fake = self.make(BGE_M3)

        vector = provider.embed_query("问题")

        self.assertEqual(fake.custom_models[0]["pooling"], provider_module.PoolingType.CLS)
        self.assertFalse(fake.custom_models[0]["normalization"])
        self.assertEqual(len(vector), 1024)
        self.assertAlmostEqual(sum(v * v for v in vector), 1.0, places=6)
        self.assertEqual(vector[:2], (0.6000000238418579, 0.800000011920929))

    def test_files_with_another_dimension_or_token_limit_are_refused(self):
        for options in ({"hidden_size": 768}, {"tokenizer_max_length": 512}):
            with self.subTest(options):
                provider_module._registered_models.clear()
                provider, _ = self.make(BGE_M3, **options)
                with self.assertRaises(EmbeddingModelMismatchError):
                    provider.embed_query("问题")


# --- Isolation: the production model is untouched ----------------------------------


class ProductionIsolationTests(unittest.TestCase):
    def test_application_cannot_be_configured_with_a_candidate(self):
        self.assertEqual(list(SUPPORTED_MODELS), ["intfloat/multilingual-e5-small"])
        self.assertEqual(DEFAULT_MODEL_NAME, "intfloat/multilingual-e5-small")
        for candidate in (E5_BASE, BGE_M3):
            with self.subTest(candidate.key), self.assertRaises(UnsupportedEmbeddingModelError):
                FastEmbedProvider(EmbeddingConfig(cache_dir=CACHE, model_name=candidate.spec.name))

    def test_production_contract_is_unchanged(self):
        self.assertEqual(EMBEDDING_VERSION, 2)
        self.assertEqual(build_contract(EmbeddingConfig(cache_dir=CACHE)), EmbeddingContract(
            model_name="intfloat/multilingual-e5-small",
            model_revision="614241f622f53c4eeff9890bdc4f31cfecc418b3", embedding_version=2,
            dimension=384, max_tokens=512, passage_prefix="passage: ", query_prefix="query: ",
            dtype="float32", normalized=True))
        self.assertEqual(model_files(MULTILINGUAL_E5_SMALL), [
            "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "onnx/model.onnx"])

    def test_baseline_is_evaluated_through_the_production_path(self):
        self.assertIs(E5_SMALL.spec, MULTILINGUAL_E5_SMALL)
        provider = make_provider(E5_SMALL, CACHE)

        self.assertEqual(provider.contract, build_contract(EmbeddingConfig(cache_dir=CACHE)))

    def test_a_spec_for_another_model_name_is_refused(self):
        config = EmbeddingConfig(cache_dir=CACHE)  # the production model name

        with self.assertRaises(UnsupportedEmbeddingModelError):
            FastEmbedProvider(config, spec=E5_BASE.spec)

    def test_vectors_of_one_model_are_never_searched_with_another(self):
        FakeRun.build()
        production = FakeRun.provider.contract
        candidate = make_provider(E5_BASE, CACHE).contract

        rows, stale = search_storage.load_candidates(FakeRun.kb.db_path, candidate)

        self.assertEqual((len(rows), stale), (0, 42))
        text = "SQLite text"
        record = EmbeddingRecord(
            chunk_id="c", model_name=production.model_name,
            model_revision=production.model_revision,
            embedding_version=production.embedding_version, dimension=production.dimension,
            max_tokens=production.max_tokens, passage_prefix=production.passage_prefix,
            dtype=production.dtype, normalized=production.normalized,
            text_sha256=text_sha256(text), input_sha256=text_sha256("passage: " + text),
            truncated=False, created_at="t")
        self.assertTrue(record.is_valid_for(production, text))
        self.assertFalse(record.is_valid_for(candidate, text))


# --- Offline -------------------------------------------------------------------------


OFFLINE_DOWNLOAD_SCRIPT = """
import os, socket, sys, tempfile
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
attempts = []
def refuse(*args, **kwargs):
    attempts.append(args)
    raise OSError("network blocked by the test")
socket.socket.connect = refuse
socket.create_connection = refuse
from app.embeddings.provider import download_model_files
from app.evaluation.candidates import E5_BASE
cache = Path(tempfile.mkdtemp())
try:
    download_model_files(E5_BASE.spec, cache)
    print("DOWNLOADED")
except Exception:
    files = [p for p in cache.rglob("*") if p.is_file() and p.suffix in (".onnx", ".json")]
    print("NETWORK" if attempts else "FILES" if files else "OFFLINE")
"""


def run_python(code: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "HF_HUB_OFFLINE"}
    finished = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env,
                              capture_output=True, text=True, timeout=120, check=False)
    return finished.stdout.strip().splitlines()[-1] if finished.stdout.strip() else finished.stderr


class OfflineTests(unittest.TestCase):
    def test_the_comparison_switches_huggingface_hub_offline_before_importing_it(self):
        output = run_python("import app.evaluation.compare_models as m, os\n"
                            "from huggingface_hub import constants\n"
                            "print(os.environ.get('HF_HUB_OFFLINE'), constants.HF_HUB_OFFLINE)")

        self.assertEqual(output, "1 True")

    def test_a_missing_model_fails_offline_without_touching_the_network(self):
        self.assertEqual(run_python(OFFLINE_DOWNLOAD_SCRIPT), "OFFLINE")

    def test_an_uncached_candidate_is_skipped_and_never_loaded(self):
        cache = Path(tempfile.mkdtemp(prefix="docubot-cache-"))
        self.addCleanup(shutil.rmtree, cache, ignore_errors=True)
        baseline = snapshot_folder(E5_SMALL.spec, cache)
        for name in model_files(E5_SMALL.spec):
            (baseline / name).parent.mkdir(parents=True, exist_ok=True)
            (baseline / name).write_bytes(b"x")
        FakeRun.build()
        runs = []

        def fake_run(key, timeout):
            runs.append(key)
            return FakeRun.result, None

        with patch.object(provider_module, "snapshot_download",
                          side_effect=AssertionError("download attempted")):
            results, skipped = compare_models.compare(["e5-small", "e5-base", "bge-m3"],
                                                      run=fake_run, cache_dir=cache)
            with self.assertRaises(compare_models.CandidateNotCachedError):
                compare_models.run_one(E5_BASE, cache)

        self.assertEqual(runs, ["e5-small"])
        self.assertEqual([r.key for r in results], ["e5-small"])
        self.assertEqual(set(skipped), {"e5-base", "bge-m3"})
        self.assertIn("not cached", skipped["e5-base"])
        self.assertEqual(len(missing_files(BGE_M3, cache)), 9)


# --- The frozen dataset and complete results -----------------------------------------


class DatasetTests(unittest.TestCase):
    def test_the_dataset_is_the_frozen_batch_6a_dataset(self):
        self.assertEqual(dataset_fingerprint(), DATASET_SHA256)
        self.assertEqual((len(PASSAGES), len(QUERIES)), (42, 33))
        self.assertEqual(sum(EXPECTED_DIRECTION_COUNTS.values()), 33)
        self.assertEqual(list(EXPECTED_DIRECTION_COUNTS), list(DIRECTIONS))

    def test_any_change_to_passages_questions_or_labels_changes_the_fingerprint(self):
        query = QUERIES[0]
        changed = {
            "passage text": ((dataclasses.replace(PASSAGES[0], text=PASSAGES[0].text + "."),
                              *PASSAGES[1:]), QUERIES),
            "question text": (PASSAGES, (dataclasses.replace(query, text=query.text + "?"),
                                         *QUERIES[1:])),
            "label": (PASSAGES, (dataclasses.replace(
                query, relevant=(Evidence(query.relevant[0].passage_key, "WAL"),)),
                *QUERIES[1:])),
            "query removed": (PASSAGES, QUERIES[1:]),
        }
        for name, (passages, queries) in changed.items():
            with self.subTest(name):
                self.assertNotEqual(dataset_fingerprint(passages, queries), DATASET_SHA256)


class ResultIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeRun.build()

    def test_a_fake_run_gives_a_complete_result_that_survives_json(self):
        result = FakeRun.result

        validate_result(result)
        self.assertEqual(ModelResult.from_json(result.to_json()), result)
        self.assertEqual(len(result.chunks), 42)
        self.assertTrue(all(len(q.top) == 5 for q in result.queries))

    def test_filtered_queries_missing_directions_and_other_datasets_are_refused(self):
        result = FakeRun.result
        broken = {
            "failed query filtered out": dataclasses.replace(
                result, queries=tuple(q for q in result.queries
                                      if (q.first_relevant_rank or 99) <= 5)),
            "direction relabelled": dataclasses.replace(result, queries=tuple(
                dataclasses.replace(q, direction="zh->zh") if q.direction == "mixed->zh" else q
                for q in result.queries)),
            "other dataset": dataclasses.replace(result, dataset_sha256="0" * 64),
            "chunk missing": dataclasses.replace(result, chunks=result.chunks[1:]),
        }
        for name, mutated in broken.items():
            with self.subTest(name), self.assertRaises(IncompleteResultError):
                validate_result(mutated)

    def test_models_must_have_searched_the_same_chunks(self):
        other = dataclasses.replace(FakeRun.result, key="e5-base", chunks=tuple(
            dataclasses.replace(c, text_sha256="0" * 64) if i == 0 else c
            for i, c in enumerate(FakeRun.result.chunks)))

        comparison.check_same_chunks([FakeRun.result, FakeRun.result])
        with self.assertRaises(IncompleteResultError):
            comparison.check_same_chunks([FakeRun.result, other])


# --- Metrics, grouping, diagnostics --------------------------------------------------


class ComparisonMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeRun.build()
        cross = [q.query_id for q in FakeRun.result.queries
                 if q.direction in CROSS_LANGUAGE_DIRECTIONS]
        cls.cross_ids = cross
        # Baseline: every cross-language question at rank 20. Candidate: first two at 1 and 4.
        cls.baseline = with_ranks(FakeRun.result, {qid: 20 for qid in cross})
        cls.candidate = dataclasses.replace(
            with_ranks(cls.baseline, {cross[0]: 1, cross[1]: 4}), key="e5-base",
            model_name="intfloat/multilingual-e5-base")

    def test_every_direction_is_reported_with_its_query_count(self):
        summaries = comparison.by_direction(FakeRun.result)

        self.assertEqual(list(summaries), list(DIRECTIONS))
        self.assertEqual({d: s.queries for d, s in summaries.items()}, EXPECTED_DIRECTION_COUNTS)

    def test_cross_language_group_is_zh_en_plus_en_zh_each_question_once(self):
        summary = comparison.cross_language(self.candidate)

        self.assertEqual(summary.queries, 10)
        self.assertEqual(len(self.cross_ids), 10)
        self.assertAlmostEqual(summary.recall[1], 0.1)
        self.assertAlmostEqual(summary.recall[5], 0.2)
        self.assertAlmostEqual(summary.mrr, (1 + 0.25 + 8 * 0.05) / 10)
        without = dataclasses.replace(self.candidate, queries=tuple(
            q for q in self.candidate.queries if q.direction != "en->zh"))
        with self.assertRaises(IncompleteResultError):
            comparison.cross_language(without)

    def test_deltas_are_candidate_minus_baseline(self):
        delta = comparison.deltas(comparison.cross_language(self.candidate),
                                  comparison.cross_language(self.baseline))

        self.assertEqual(list(delta), ["R@1", "R@3", "R@5", "MRR"])
        self.assertAlmostEqual(delta["R@1"], 0.1)
        self.assertAlmostEqual(delta["R@5"], 0.2)
        self.assertAlmostEqual(delta["MRR"], (1 - 0.05 + 0.25 - 0.05) / 10)
        with self.assertRaises(ValueError):
            comparison.deltas(summarize([QueryOutcome("q", "en->en", (1,))]),
                              comparison.cross_language(self.baseline))

    def test_profile_separates_rank_1_successes_better_ranks_and_language_mix(self):
        profile = comparison.cross_language_profile(self.candidate)

        self.assertEqual(profile.first_ranks[:3], (1, 4, 20))
        self.assertEqual((profile.at_rank_1, profile.in_top_3, profile.in_top_5), (1, 1, 2))
        self.assertEqual(profile.top_slots, 50)
        expected_same = sum(hit.language == q.query_language for q in self.candidate.queries
                            if q.direction in CROSS_LANGUAGE_DIRECTIONS for hit in q.top)
        self.assertEqual(profile.query_language_in_top, expected_same)

    def test_truncation_is_counted_per_chunk_language(self):
        chunks = FakeRun.result.chunks
        chinese = [i for i, c in enumerate(chunks) if c.language == "zh"][:3]
        result = dataclasses.replace(FakeRun.result, chunks=tuple(
            dataclasses.replace(c, truncated=i in chinese) for i, c in enumerate(chunks)))

        table = comparison.truncation(result)

        self.assertEqual((table["zh"].truncated, table["zh"].chunks), (3, 18))
        self.assertEqual((table["en"].truncated, table["mixed"].truncated), (0, 0))
        self.assertEqual((table["total"].truncated, table["total"].chunks), (3, 42))
        self.assertAlmostEqual(table["zh"].percent, 100 * 3 / 18)

    def test_diagnostics_list_every_cross_language_question_with_its_top_5(self):
        text = comparison_report.diagnostics_section([self.candidate])
        golden = {q.query_id: q for q in QUERIES}

        for query_id in self.cross_ids:
            self.assertIn(golden[query_id].text, text)
            self.assertIn(golden[query_id].relevant[0].text, text)
        others = [q.query_id for q in QUERIES if q.query_id not in self.cross_ids]
        self.assertFalse(any(f"| {query_id} |" in text for query_id in others))
        first = self.candidate.queries[0]
        self.assertTrue(all(hit.relevant == (hit.chunk_id in runner.relevant_chunk_ids(
            FakeRun.kb, QUERIES[0])) for hit in first.top))

    def test_full_report_shows_every_model_direction_and_signed_delta(self):
        report = comparison_report.render([self.baseline, self.candidate],
                                          {"bge-m3": "Candidate B skipped due to measured resource "
                                                     "constraints: test"}, {"python": "3"}, 60)

        for heading in ("## Overall", "## Cross-language (primary metric)",
                        "## Language breakdown", "## Truncation", "## Resources",
                        "## Per-query cross-language diagnostics"):
            self.assertIn(heading, report)
        for direction in DIRECTIONS:
            self.assertIn(f"| {direction} |", report)
        self.assertIn("+0.100", report)
        self.assertIn("Candidate B skipped", report)


# --- Checks of a run and the resource gate --------------------------------------------


class RunCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeRun.build()

    def expected_inputs(self, prefix_of_passages="passage: ", prefix_of_queries="query: "):
        return ([prefix_of_passages + c.text for c in FakeRun.kb.chunks.values()]
                + [prefix_of_queries + q.text for q in QUERIES])

    def test_correct_inputs_pass_the_input_contract_checks(self):
        checks = compare_models.input_checks(FakeRun.provider, FakeRun.kb, self.expected_inputs())

        self.assertEqual(set(checks.values()), {True})

    def test_swapped_missing_or_extra_prefixes_fail_the_input_checks(self):
        cases = {
            "swapped": self.expected_inputs("query: ", "passage: "),
            "passage prefix removed": self.expected_inputs(prefix_of_passages=""),
            "query prefix removed": self.expected_inputs(prefix_of_queries=""),
        }
        for name, seen in cases.items():
            with self.subTest(name):
                checks = compare_models.input_checks(FakeRun.provider, FakeRun.kb, seen)
                self.assertIn(False, checks.values())

    def test_vector_checks_catch_unnormalized_and_nondeterministic_vectors(self):
        good = compare_models.vector_checks(FakeRun.provider, FakeRun.kb)
        self.assertEqual(set(good.values()), {True})

        unnormalized = FakeEmbeddingProvider(normalize=False)
        self.assertFalse(compare_models.vector_checks(unnormalized, FakeRun.kb)["normalized"])

        class Drifting(FakeEmbeddingProvider):
            """Returns another vector for the same text on every call."""

            calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                return super().embed_documents([text + "#" * self.calls for text in texts])

        self.assertFalse(compare_models.vector_checks(Drifting(), FakeRun.kb)
                         ["deterministic_vectors"])

    def test_optional_model_is_skipped_when_its_measured_estimate_does_not_fit(self):
        small = dataclasses.replace(FakeRun.result, resources=resources(
            download_bytes=int(0.5 * GB), peak_private_bytes=int(1.5 * GB)))
        base = dataclasses.replace(FakeRun.result, key="e5-base", resources=resources(
            download_bytes=int(1.0 * GB), peak_private_bytes=int(2.0 * GB)))
        estimate = compare_models.estimate_peak_bytes(BGE_M3, [small, base])
        self.assertAlmostEqual(estimate / GB, 2.0 + (BGE_M3.download_bytes / GB - 1.0), places=3)

        tight = SystemMemory(16 * GB, int(2 * GB), int(8 * GB))
        roomy = SystemMemory(16 * GB, int(8 * GB), int(8 * GB))
        reason = compare_models.resource_skip_reason(BGE_M3, [small, base], tight)
        self.assertIn("Candidate B skipped due to measured resource constraints", reason)
        self.assertIsNone(compare_models.resource_skip_reason(BGE_M3, [small, base], roomy))
        self.assertIsNone(compare_models.estimate_peak_bytes(BGE_M3, [small]))


if __name__ == "__main__":
    unittest.main()
