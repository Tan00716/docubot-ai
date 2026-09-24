"""Smoke test with the REAL local model (opt-in, not part of the normal run).

It loads intfloat/multilingual-e5-small (pinned revision) from
storage/model_cache/ (downloading it once, ~490 MB, if it is not there yet),
embeds English, Chinese and mixed texts on the CPU, checks the E5
"passage: " / "query: " input contract against the raw model, re-computes
one vector independently (ONNX Runtime + mean pooling, the model card's
recipe) and stores and reloads vectors in a temporary database.

This is an embedding-CONTRACT smoke test. It does NOT judge retrieval quality.

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
from app.embeddings.config import DEFAULT_MODEL_NAME, MULTILINGUAL_E5_SMALL, EmbeddingConfig  # noqa: E402
from app.embeddings.provider import FastEmbedProvider, download_model_files  # noqa: E402
from app.embeddings.vectors import l2_normalize, text_sha256, to_float32  # noqa: E402
from test_embedding_storage import EmbeddingTestCase  # noqa: E402

SHORT_EN = "FastAPI is a Python web framework."
SHORT_ZH = "FastAPI 是一个 Python Web 框架。"
MEDIUM_EN = (
    "DocuBot splits every uploaded document into chunks of about 1200 characters. "
    "Each chunk is turned into a vector by a local embedding model that runs on the CPU. "
    "The vectors are stored in SQLite together with the exact contract that produced them, "
    "so vectors from different models or settings are never compared with each other."
)
MEDIUM_ZH = (
    "DocuBot 会把每个上传的文档切分成大约一千二百个字符的文本块。"
    "每个文本块都由在 CPU 上本地运行的嵌入模型转换成一个向量。"
    "向量和生成它的嵌入契约一起保存在 SQLite 中，因此不同模型或不同设置产生的向量永远不会被放在一起比较。"
)
MIXED_1200 = (
    "Embedding Model（嵌入模型）把一段文字变成一组数字，也就是 vector（向量）。"
    "For example, the sentence 'FastAPI is a Python web framework' becomes 384 float32 numbers. "
    "Max Sequence Length（最大序列长度）是模型一次最多能读取的 token 数量；"
    "multilingual-e5-small reads at most 512 tokens, and anything after that is cut off. "
    "这种截断叫 Token Truncation，被截掉的内容不会影响向量。"
    "Passage embeddings are used for document chunks and start with 'passage: '. "
    "Query embeddings are used for questions and start with 'query: '. "
    "如果把两个前缀用反，程序不会报错，但检索效果会变差，所以需要测试来保护这个约定。"
    "The embedding contract records the model name, the model revision, the embedding version, "
    "the dimension, the token limit, the passage prefix, the dtype and the normalization. "
    "只要其中任何一项发生变化，旧的向量就变成 Stale Vector（过期向量），必须重新生成。"
    "In SQLite every chunk keeps exactly one active embedding, and deleting a chunk also "
    "deletes its embedding through ON DELETE CASCADE. "
    "Model Revision 是 Hugging Face 仓库的 git commit hash，固定它可以保证每次下载的模型文件完全相同。"
    "Batch size decides how many chunks are sent to the model at once: too large uses a lot "
    "of memory, too small is slower. 在 Intel i5-1334U 这样的笔记本 CPU 上，16 是比较稳妥的默认值。"
    "All normal tests run without network access; this real-model smoke test must be switched on "
    "explicitly, because it loads a model of about 470 MB into memory before it can run."
)[:1200]
LONG_ZH = (
    "检索增强生成是一种把信息检索和大语言模型结合起来的技术。系统首先把用户上传的文档切分成较小的文本块，"
    "然后使用嵌入模型把每个文本块转换成一个向量。语义相近的文本在向量空间中的距离也比较近。"
    "当用户提出问题时，系统用同一个嵌入模型把问题转换成向量，并查找最相似的文本块，"
    "再把这些文本块作为证据交给大语言模型生成回答，同时附上引用来源。"
) * 8
QUERY = "What is FastAPI?"
DOCUMENTS = [SHORT_EN, SHORT_ZH, MEDIUM_EN, MEDIUM_ZH, MIXED_1200]


def model_is_cached(cache_dir: Path) -> bool:
    snapshot = (cache_dir / "models--intfloat--multilingual-e5-small" / "snapshots"
                / MULTILINGUAL_E5_SMALL.revision)
    return (snapshot / MULTILINGUAL_E5_SMALL.model_file).is_file()


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


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
            cls.results = cls.provider.embed_documents(DOCUMENTS)
            cls.model_dir = download_model_files(cls.provider.spec, cache_dir)
        cls.load_seconds = time.perf_counter() - started
        cls.offline = bool(offline)
        cls.raw_model = cls.provider._get_model()

    def raw(self, model_input):
        """The model's own output for an exact input, normalized like DocuBot does."""
        values = next(iter(self.raw_model.embed([model_input], batch_size=1))).tolist()
        return to_float32(l2_normalize(values))

    def assert_same_vector(self, first, second):
        self.assertGreater(dot(first, second), 0.99999)

    def test_model_loads_locally_with_the_pinned_contract(self):
        print(f"\n  model load + first batch: {self.load_seconds:.1f}s "
              f"(offline={self.offline})", file=sys.stderr)
        contract = self.provider.contract
        self.assertEqual((contract.model_name, contract.model_revision),
                         (DEFAULT_MODEL_NAME, "614241f622f53c4eeff9890bdc4f31cfecc418b3"))
        self.assertEqual((contract.dimension, contract.max_tokens), (384, 512))
        self.assertEqual(self.raw_model.model.tokenizer.truncation["max_length"], 512)
        self.assertEqual(self.model_dir.name, contract.model_revision)

    def test_documents_give_384_dimensional_unit_vectors_without_truncation(self):
        for text, result in zip(DOCUMENTS, self.results):
            with self.subTest(text=text[:30]):
                self.assertEqual(len(result.vector), 384)
                self.assertAlmostEqual(math.sqrt(dot(result.vector, result.vector)), 1.0,
                                       places=5)
                self.assertFalse(result.truncated)
                self.assertEqual(result.input_sha256, text_sha256("passage: " + text))

    def test_document_vectors_are_exactly_passage_inputs(self):
        for text, result in zip(DOCUMENTS, self.results):
            with self.subTest(text=text[:30]):
                self.assert_same_vector(result.vector, self.raw("passage: " + text))
                self.assertLess(dot(result.vector, self.raw(text)), 0.9999,
                                "the prefix must change the input")

    def test_query_vector_is_exactly_the_query_input(self):
        query = self.provider.embed_query(QUERY)

        self.assertEqual(len(query), 384)
        self.assertAlmostEqual(math.sqrt(dot(query, query)), 1.0, places=5)
        self.assert_same_vector(query, self.raw("query: " + QUERY))
        self.assertLess(dot(query, self.raw("passage: " + QUERY)), 0.9999,
                        "embed_query must not use the passage prefix")

    def test_query_and_document_share_one_embedding_space(self):
        # A sanity check of the contract, NOT a retrieval-quality measurement.
        query = self.provider.embed_query(QUERY)
        [related, unrelated] = self.provider.embed_documents(
            [SHORT_EN, "新加坡位于马来半岛南端，属于热带雨林气候。"])
        print(f"\n  cos(query, related)={dot(query, related.vector):.3f} "
              f"cos(query, unrelated)={dot(query, unrelated.vector):.3f}", file=sys.stderr)

        self.assertGreater(dot(query, related.vector), dot(query, unrelated.vector))

    def test_vector_matches_an_independent_mean_pooling_implementation(self):
        # The model card's recipe: tokenize (max 512), run the ONNX model,
        # average the token vectors where attention_mask = 1, then L2-normalize.
        import numpy
        import onnxruntime
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=512)
        session = onnxruntime.InferenceSession(str(self.model_dir / "onnx" / "model.onnx"),
                                               providers=["CPUExecutionProvider"])
        encoding = tokenizer.encode("passage: " + MEDIUM_ZH)
        feeds = {"input_ids": [encoding.ids], "attention_mask": [encoding.attention_mask],
                 "token_type_ids": [encoding.type_ids]}
        feeds = {i.name: numpy.array(feeds[i.name], dtype=numpy.int64)
                 for i in session.get_inputs()}
        hidden = session.run(None, feeds)[0][0]
        mask = numpy.array(encoding.attention_mask, dtype=numpy.float32)[:, None]
        pooled = (hidden * mask).sum(axis=0) / mask.sum()
        reference = to_float32(l2_normalize(pooled.tolist()))

        self.assert_same_vector(self.results[DOCUMENTS.index(MEDIUM_ZH)].vector, reference)

    def test_same_text_gives_the_same_vector(self):
        again = self.provider.embed_documents([SHORT_EN])[0].vector

        self.assert_same_vector(again, self.results[0].vector)

    def test_long_chinese_chunk_is_flagged_as_truncated(self):
        # Known limitation: Chinese needs about 0.64 tokens per character, so a
        # full 1200-character Chinese chunk does not fit into 512 tokens.
        chunk = LONG_ZH[:1200]
        [result] = self.provider.embed_documents([chunk])

        self.assertEqual(len(chunk), 1200)
        self.assertTrue(result.truncated)

    def test_real_vectors_are_stored_and_loaded(self):
        document_id = self.chunked_document(
            "\n\n".join(DOCUMENTS).encode("utf-8"), config=None)

        started = time.perf_counter()
        summary = service.embed_document(document_id, upload_dir=self.upload_dir,
                                         db_path=self.db_path, provider=self.provider)
        print(f"\n  embedded {summary.embedded_count} chunk(s) in "
              f"{time.perf_counter() - started:.2f}s", file=sys.stderr)

        status = service.get_embedding_status(document_id, upload_dir=self.upload_dir,
                                              db_path=self.db_path,
                                              contract=self.provider.contract)
        self.assertEqual(status.embedded_chunks, status.total_chunks)
        for chunk in self.chunks(document_id):
            stored = storage.get_embedding(self.db_path, chunk.chunk_id)
            self.assertFalse(chunk.text.startswith("passage:"))
            self.assertEqual((stored.record.dimension, stored.record.passage_prefix,
                              stored.record.model_revision),
                             (384, "passage: ", MULTILINGUAL_E5_SMALL.revision))
            self.assertEqual(stored.record.input_sha256, text_sha256("passage: " + chunk.text))
            self.assert_same_vector(stored.vector, self.raw("passage: " + chunk.text))


if __name__ == "__main__":
    unittest.main()
