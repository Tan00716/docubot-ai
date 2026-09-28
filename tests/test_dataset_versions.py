"""Integrity and negative tests for versioned retrieval evaluation data."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.evaluation import compare_models, comparison
from app.evaluation.analysis import fingerprint
from app.evaluation.corpus import LANGUAGES
from app.evaluation.dataset_versions import DATASET_DIR, VERSIONS, verify_dataset, write_artifact
from app.evaluation.expansion_v2 import HARD_NEGATIVES_V2
from app.evaluation.golden import validate_dataset
from app.evaluation.resources import SystemMemory
from app.evaluation.candidates import get_candidate
from app.evaluation.relevance_review import ReviewError, finalize_blind_reviews


class DatasetVersionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.v1 = verify_dataset("retrieval_eval_v1")
        cls.v2 = verify_dataset("retrieval_eval_v2")
        cls.corpus = {p.key: p for p in cls.v2.passages}
        cls.queries = {q.query_id: q for q in cls.v2.queries}

    def test_both_versions_are_known(self):
        self.assertEqual(VERSIONS, ("retrieval_eval_v1", "retrieval_eval_v2"))

    def test_v1_retains_the_original_42_passages_and_33_queries(self):
        self.assertEqual((len(self.v1.passages), len(self.v1.queries)), (42, 33))

    def test_v1_keeps_the_batch_6b_fingerprint_contract(self):
        self.assertEqual(comparison.dataset_fingerprint(), comparison.DATASET_SHA256)

    def test_v2_is_expanded_without_mutating_v1(self):
        self.assertEqual((len(self.v2.passages), len(self.v2.queries)), (78, 70))
        self.assertEqual(tuple(p.key for p in self.v2.passages[:42]),
                         tuple(p.key for p in self.v1.passages))
        self.assertEqual(tuple(q.query_id for q in self.v2.queries[:33]),
                         tuple(q.query_id for q in self.v1.queries))

    def test_v2_artifact_checksum_is_verified(self):
        self.assertEqual(verify_dataset("retrieval_eval_v2").sha256, self.v2.sha256)

    def test_artifact_checksum_rejects_a_changed_query(self):
        self._mutated_artifact(lambda d: d["queries"][0].update(text="changed"))

    def test_artifact_checksum_rejects_a_changed_passage(self):
        self._mutated_artifact(lambda d: d["corpus"][0].update(text="changed"))

    def test_artifact_checksum_rejects_a_changed_relevance_label(self):
        self._mutated_artifact(lambda d: d["queries"][0]["relevant"][0].update(evidence="changed"))

    def test_artifact_checksum_rejects_a_removed_hard_negative(self):
        self._mutated_artifact(lambda d: d["hard_negatives"].pop(next(iter(d["hard_negatives"]))))

    def test_artifact_checksum_rejects_a_changed_version(self):
        self._mutated_artifact(lambda d: d.update(version="retrieval_eval_v9"))

    def test_existing_dataset_artifact_cannot_be_overwritten_implicitly(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "retrieval_eval_v1.json"
            shutil.copyfile(DATASET_DIR / path.name, path)
            with self.assertRaises(FileExistsError):
                write_artifact("retrieval_eval_v1", Path(root))

    def test_v2_labels_are_valid(self):
        validate_dataset(self.v2.passages, self.v2.queries)

    def test_v2_language_distribution_is_balanced(self):
        counts = {lang: sum(p.language == lang for p in self.v2.passages) for lang in LANGUAGES}
        self.assertEqual(counts, {"en": 30, "zh": 32, "mixed": 16})

    def test_query_language_distribution_is_present(self):
        counts = {lang: sum(q.language == lang for q in self.v2.queries) for lang in LANGUAGES}
        self.assertTrue(all(counts.values()))

    def test_every_required_evaluation_direction_is_present(self):
        directions = {comparison.direction_of(q, self.v2.passages) for q in self.v2.queries}
        self.assertTrue({"en->en", "zh->zh", "zh->en", "en->zh", "mixed->en", "mixed->zh"}
                        <= directions)

    def test_cross_language_count_is_materially_above_batch_6b(self):
        cross = [q for q in self.v2.queries
                 if q.language != comparison.direction_of(q, self.v2.passages).split("->")[1]]
        self.assertEqual(len(cross), 36)
        zh_en = sum(q.language == "zh" and comparison.direction_of(q, self.v2.passages).endswith("->en")
                    for q in cross)
        en_zh = sum(q.language == "en" and comparison.direction_of(q, self.v2.passages).endswith("->zh")
                    for q in cross)
        self.assertEqual((zh_en, en_zh), (12, 12))
        self.assertGreater(zh_en + en_zh, 2 * 10)

    def test_every_new_query_has_a_hard_negative(self):
        new_ids = {q.query_id for q in self.v2.queries[33:]}
        self.assertEqual(set(HARD_NEGATIVES_V2), new_ids)
        self.assertTrue(all(HARD_NEGATIVES_V2.values()))

    def test_hard_negatives_name_existing_passages(self):
        for query_id, keys in HARD_NEGATIVES_V2.items():
            with self.subTest(query=query_id):
                self.assertTrue(set(keys) <= self.corpus.keys())

    def test_hard_negative_is_not_labeled_relevant_for_its_query(self):
        for query_id, keys in HARD_NEGATIVES_V2.items():
            relevant = {e.passage_key for e in self.queries[query_id].relevant}
            with self.subTest(query=query_id):
                self.assertFalse(relevant & set(keys))

    def test_long_chinese_faq_exceeds_the_character_estimate_for_512_tokens(self):
        self.assertGreater(len(self.corpus["v2_zh_long_faq"].text), 790)

    def test_long_faq_has_early_evidence(self):
        self.assertIn("无效字段会得到 422 响应", self.corpus["v2_zh_long_faq"].text)

    def test_long_faq_has_middle_evidence(self):
        text = self.corpus["v2_zh_long_faq"].text
        self.assertGreater(text.index("采用指数退避和随机抖动"), len(text) // 3)

    def test_long_faq_has_late_evidence(self):
        text = self.corpus["v2_zh_long_faq"].text
        self.assertGreater(text.index("持久数据库文件应放在挂载卷"), len(text) * 2 // 3)

    def test_three_queries_cover_early_middle_and_late_faq_evidence(self):
        ids = {"v2_zh_zh_faq_begin", "v2_zh_zh_faq_middle", "v2_zh_zh_faq_end"}
        self.assertEqual({q.query_id for q in self.v2.queries if q.query_id in ids}, ids)
        self.assertEqual({self.queries[q].relevant[0].passage_key for q in ids},
                         {"v2_zh_long_faq"})

    def test_v2_contains_all_three_query_languages(self):
        self.assertEqual({q.language for q in self.v2.queries}, set(LANGUAGES))

    def test_invalid_version_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_dataset("retrieval_eval_v3")

    def test_changed_source_record_is_rejected_against_the_artifact(self):
        from app.evaluation import dataset_versions
        original = dataset_versions.PASSAGES_V2
        try:
            dataset_versions.PASSAGES_V2 = original[:-1]
            with self.assertRaises(ValueError):
                verify_dataset("retrieval_eval_v2")
        finally:
            dataset_versions.PASSAGES_V2 = original

    def _mutated_artifact(self, mutation):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root)
            shutil.copyfile(DATASET_DIR / "retrieval_eval_v2.json",
                            destination / "retrieval_eval_v2.json")
            path = destination / "retrieval_eval_v2.json"
            artifact = json.loads(path.read_text(encoding="utf-8"))
            mutation(artifact)
            path.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_dataset("retrieval_eval_v2", destination)


class ModelSelectionTests(unittest.TestCase):
    def test_skip_bge_leaves_the_e5_candidates(self):
        self.assertEqual(compare_models.selected_model_keys(
            ["e5-small", "e5-base", "bge-m3"], skip_bge=True), ["e5-small", "e5-base"])

    def test_only_bge_selects_only_bge(self):
        self.assertEqual(compare_models.selected_model_keys(
            ["e5-small", "e5-base"], only_bge=True), ["bge-m3"])

    def test_skip_bge_and_only_bge_produces_an_empty_selection(self):
        self.assertEqual(compare_models.selected_model_keys(
            ["e5-small"], skip_bge=True, only_bge=True), [])

    def test_bge_is_skipped_when_available_memory_is_below_its_safe_reserve(self):
        memory = SystemMemory(16 * 1024**3, int(2.17 * 1024**3), int(3.15 * 1024**3))
        reason = compare_models.resource_preflight(get_candidate("bge-m3"), memory)
        self.assertIn("NOT MEASURED — insufficient safe memory", reason)

    def test_e5_base_is_skipped_when_its_prior_peak_does_not_fit(self):
        memory = SystemMemory(16 * 1024**3, int(2.17 * 1024**3), int(3.15 * 1024**3))
        reason = compare_models.resource_preflight(get_candidate("e5-base"), memory)
        self.assertIn("NOT MEASURED — insufficient safe memory", reason)

    def test_unknown_memory_is_treated_as_unsafe(self):
        reason = compare_models.resource_preflight(get_candidate("bge-m3"),
                                                   SystemMemory(None, None, None))
        self.assertIsNotNone(reason)


class BlindReviewWorkflowTests(unittest.TestCase):
    def test_explicitly_ambiguous_query_is_removed_from_golden_candidates(self):
        accepted, removed = finalize_blind_reviews([{
            "query_id": "q1", "ambiguous": True, "candidates": [], "judgments": []
        }])
        self.assertEqual(accepted, [])
        self.assertEqual(removed, ["q1"])

    def test_every_candidate_requires_one_review_judgment(self):
        with self.assertRaises(ReviewError):
            finalize_blind_reviews([{
                "query_id": "q1", "ambiguous": False,
                "candidates": [{"candidate_id": "c1"}, {"candidate_id": "c2"}],
                "judgments": [{"candidate_id": "c1", "label": "RELEVANT"}],
            }])

    def test_binary_metric_treats_partial_relevance_as_not_relevant(self):
        records, removed = finalize_blind_reviews([{
            "query_id": "q1", "ambiguous": False,
            "candidates": [{"candidate_id": "c1"}, {"candidate_id": "c2"}],
            "judgments": [{"candidate_id": "c1", "label": "RELEVANT"},
                          {"candidate_id": "c2", "label": "PARTIALLY_RELEVANT"}],
        }])
        self.assertEqual(len(records), 1)
        self.assertEqual(removed, [])

    def test_query_with_no_relevant_evidence_must_be_removed_as_ambiguous(self):
        with self.assertRaises(ReviewError):
            finalize_blind_reviews([{
                "query_id": "q1", "ambiguous": False,
                "candidates": [{"candidate_id": "c1"}],
                "judgments": [{"candidate_id": "c1", "label": "NOT_RELEVANT"}],
            }])

    def test_unknown_review_label_is_rejected(self):
        with self.assertRaises(ReviewError):
            finalize_blind_reviews([{
                "query_id": "q1", "ambiguous": False,
                "candidates": [{"candidate_id": "c1"}],
                "judgments": [{"candidate_id": "c1", "label": "MAYBE"}],
            }])

    def test_ranking_mutation_changes_the_determinism_fingerprint(self):
        query = SimpleNamespace(query_id="q1")
        outcome = SimpleNamespace(ranks=(1,))
        common = {"query": query, "outcome": outcome, "secondary_ranks": (),
                  "truncation_group": "not truncated"}
        first = SimpleNamespace(**(common | {"ranking": (("chunk-a", 0.9), ("chunk-b", 0.8))}))
        mutated = SimpleNamespace(**(common | {"ranking": (("chunk-b", 0.9), ("chunk-a", 0.8))}))
        self.assertNotEqual(fingerprint([first]), fingerprint([mutated]))


if __name__ == "__main__":
    unittest.main()
