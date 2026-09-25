"""Golden relevance labels: evaluation questions and the evidence that answers them.

A label does not name a chunk ID (chunk IDs change with the chunk size). It
names a passage and an EVIDENCE sentence: an exact piece of that passage
that answers the question. After chunking, every chunk containing the whole
evidence text is a relevant chunk for that label.

    relevant  - the evidence a good search must find. Each label is ONE
                relevant item, even if (because of chunk overlap) two chunks
                contain it: Recall@K counts items, not chunks.
    secondary - acceptable, partly relevant evidence. It never counts for
                Recall@K or MRR; it only helps to explain failures (was the
                top result wrong, or merely second best?).

The questions are written the way a user would ask, avoiding the wording of
the answer (a guard test checks that no long phrase is copied from the
answer passage). The direction of a question is "<question language> ->
<answer language>", e.g. "zh->en" = Chinese question, English answer.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.corpus import LANGUAGES, PASSAGES, Language, Passage


@dataclass(frozen=True)
class Evidence:
    passage_key: str
    text: str  # an exact substring of the passage


@dataclass(frozen=True)
class GoldenQuery:
    query_id: str
    language: Language  # language of the question itself
    text: str
    relevant: tuple[Evidence, ...]
    secondary: tuple[Evidence, ...] = ()


def _q(query_id: str, language: Language, text: str, relevant: list[tuple[str, str]],
       secondary: list[tuple[str, str]] | None = None) -> GoldenQuery:
    return GoldenQuery(
        query_id=query_id,
        language=language,
        text=text,
        relevant=tuple(Evidence(key, evidence) for key, evidence in relevant),
        secondary=tuple(Evidence(key, evidence) for key, evidence in secondary or []),
    )


# Report order. The first six are the directions this evaluation must cover;
# the last three target the mixed-language passages.
DIRECTIONS = (
    "en->en", "zh->zh", "zh->en", "en->zh", "mixed->en", "mixed->zh",
    "zh->mixed", "en->mixed", "mixed->mixed",
)

QUERIES: tuple[GoldenQuery, ...] = (
    # --- English question -> English answer ----------------------------------
    _q("en_en_sqlite_wal", "en",
       "Will lookups stall while another request is saving data into the same local database "
       "file?",
       [("en_sqlite_wal", "write-ahead logging")]),
    _q("en_en_prefix", "en",
       "What happens to search quality if the model is never told whether a text is a question "
       "or a document?",
       [("en_e5_prefix", "measurably worse at finding the right paragraph")]),
    _q("en_en_docker", "en",
       "Why does rebuilding my container image download every Python package again after I "
       "edit one source file?",
       [("en_docker_layers", "invalidates the cache")]),
    _q("en_en_rate_limit", "en",
       "My client is being rejected for firing too many calls in a short time. How does the "
       "server indicate how long to back off?",
       [("en_rate_limit", "Retry-After")]),
    _q("en_en_rag_diagnosis", "en",
       "How can I tell whether a wrong chatbot answer was caused by the search step or by the "
       "language model?",
       [("en_rag_failure_types", "the problem is retrieval")],
       [("zh_rag_hallucination", "最好把检索和生成分开检查")]),
    # --- Chinese -> Chinese ----------------------------------------------------
    _q("zh_zh_normalize", "zh",
       "把向量缩放到单位长度，对计算相似度有什么好处？",
       [("zh_normalize_cosine", "余弦相似度就等于点积本身")]),
    _q("zh_zh_status_code", "zh",
       "接口收到的参数类型不对时，应该返回哪个状态码？",
       [("zh_http_422_400", "422 则表示格式可以解析")]),
    _q("zh_zh_container_data", "zh",
       "把容器删掉以后，之前存在里面的数据还能找回来吗？",
       [("zh_docker_volume", "容器一旦被删除")]),
    _q("zh_zh_ties", "zh",
       "两个结果得分一模一样时，怎么保证每次返回的顺序都相同？",
       [("zh_deterministic_ties", "分数相同时按块的编号从小到大排列")],
       [("zh_exact_search", "分数完全相同的情况再按块的编号排序")]),
    # The next two answers sit at the END of long Chinese passages: with
    # 1200-character chunks they lie beyond the model's 512-token window.
    _q("zh_zh_ann_threshold", "zh",
       "数据量涨到多大时，才有必要改用专门的向量索引？",
       [("zh_exact_search", "当块的数量达到数十万甚至上百万")]),
    _q("zh_zh_log_retention", "zh",
       "调试用的日志一般留多长时间就该清理？",
       [("zh_logging_privacy", "调试日志通常保留几天到几周即可")]),
    # Three answers from one long FAQ: the first is inside the model's window,
    # the other two lie beyond it at 1200/200, and the visible part of the
    # FAQ is about other topics (file types, sizes, failures, ...).
    _q("zh_zh_faq_doc_format", "zh",
       "用 Word 2003 保存的 .doc 文件能不能上传？",
       [("zh_faq", "旧版 Word 的 DOC 格式、图片和压缩包都会被拒绝")]),
    _q("zh_zh_faq_storage", "zh",
       "我的文件是存在自己电脑上，还是会被发到外部服务器？",
       [("zh_faq", "不会发送到任何云端服务")]),
    _q("zh_zh_faq_telegram", "zh",
       "现在能不能在 Telegram 里直接问文档里的问题？",
       [("zh_faq", "还不会根据文档内容回答问题")]),
    # --- Chinese -> English ----------------------------------------------------
    _q("zh_en_overlap", "zh",
       "为什么相邻的两个文本片段之间要重复一小段内容？",
       [("en_overlap", "copies the last part of each piece to the beginning of the following one")],
       [("zh_chunk_size_tradeoff", "重叠区域的长度通常随块的大小一起调整")]),
    _q("zh_en_postgres", "zh",
       "哪种数据库作为独立的服务进程运行，并且允许很多客户端同时写入？",
       [("en_postgres", "many clients can write at the same time")]),
    _q("zh_en_metrics", "zh",
       "怎么衡量正确的资料在搜索结果中是不是排得足够靠前？",
       [("en_eval_metrics", "Mean reciprocal rank looks at the position of the first relevant "
                            "result")]),
    _q("zh_en_upload_name", "zh",
       "用户上传文件时，怎样防止文件名把文件写到别的目录里去？",
       [("en_upload_path", "never uses the client's name on disk")]),
    _q("zh_en_vector_bytes", "zh",
       "每个向量存进数据库以后大概占多少字节？",
       [("en_float32_storage", "1536 bytes")],
       [("zh_exact_search", "每个块的向量占用约一点五千字节")]),
    # --- English -> Chinese ----------------------------------------------------
    _q("en_zh_injection", "en",
       "How do I stop text typed by a user from being executed as part of a database command?",
       [("zh_sql_injection", "使用参数化查询")]),
    _q("en_zh_token_leak", "en",
       "My bot's secret credential ended up in a public repository. Is deleting the commit "
       "enough?",
       [("zh_bot_token", "仅仅删除那次提交是不够的")]),
    _q("en_zh_hallucination", "en",
       "Why does a language model sometimes state things that are not in the retrieved "
       "material, and how can such claims be checked?",
       [("zh_rag_hallucination", "这种现象通常称为幻觉")],
       [("en_rag_failure_types", "the problem is generation")]),
    _q("en_zh_chunk_size", "en",
       "What goes wrong when the pieces a document is split into are very large or very small?",
       [("zh_chunk_size_tradeoff", "块太大时，一个块里混杂了好几个主题")]),
    _q("en_zh_cjk_tokens", "en",
       "Why does Chinese text fill up the model's input limit faster than English text of the "
       "same length?",
       [("zh_tokenization_cjk", "中文却可能有六七百个")],
       [("mixed_truncation_512", "中文 chunk 更容易超出")]),
    # --- Mixed question -> English answer --------------------------------------
    _q("mixed_en_polling", "mixed",
       "Telegram bot 用 long polling 的时候，是怎么拿到新消息的？",
       [("en_telegram_polling", "repeatedly calls getUpdates")],
       [("mixed_webhook_https", "除了 long polling")]),
    _q("mixed_en_validation", "mixed",
       "request body 里 top_k 传的是字符串 \"5\"，FastAPI 会怎么处理？",
       [("en_fastapi_validation", "is not silently converted to the integer 5")],
       [("zh_http_422_400", "422 则表示格式可以解析")]),
    _q("mixed_en_ann", "mixed",
       "FAISS 或者 HNSW 这类 index 为什么有时会漏掉最相似的结果？",
       [("en_ann_index", "a truly closest vector can occasionally be skipped")]),
    # --- Mixed question -> Chinese answer --------------------------------------
    _q("mixed_zh_batch", "mixed",
       "embedding 的 batch size 设得太大会有什么问题？",
       [("zh_batch_memory", "批次过大可能让内存占用突然升高")]),
    _q("mixed_zh_pdf_table", "mixed",
       "从 PDF 里提取出来的 table 顺序全乱了，是什么原因？",
       [("zh_pdf_extraction", "表格的单元格也常常按坐标顺序连成一行")]),
    _q("mixed_zh_log_question", "mixed",
       "log 里可以直接记录 user 的原始 question 吗？",
       [("zh_logging_privacy", "而不记录问题原文和返回的文本内容")]),
    # --- Questions whose answer is a mixed-language passage --------------------
    _q("zh_mixed_shared_model", "zh",
       "怎样让所有请求共用同一个已经加载好的模型？",
       [("mixed_fastapi_depends", "只会真正创建一次 provider")]),
    _q("en_mixed_async", "en",
       "Should a slow, CPU-heavy model call go into an async endpoint?",
       [("mixed_async_endpoints", "整个 event loop 都会被卡住")]),
    _q("mixed_mixed_long_query", "mixed",
       "如果用户的问题特别长，超出 tokenizer 能处理的长度，系统应该怎么办？",
       [("mixed_truncation_512", "更合理的做法是直接拒绝过长的 query")]),
)


# --- Integrity ----------------------------------------------------------------


def direction_of(query: GoldenQuery, passages: Sequence[Passage] = PASSAGES) -> str:
    """ "<question language>-><answer language>"; all relevant passages share one language."""
    language_of = {passage.key: passage.language for passage in passages}
    targets = {language_of[evidence.passage_key] for evidence in query.relevant}
    if len(targets) != 1:
        raise ValueError(f"Query {query.query_id!r} must target exactly one answer language.")
    return f"{query.language}->{targets.pop()}"


def validate_dataset(passages: Sequence[Passage] = PASSAGES,
                     queries: Sequence[GoldenQuery] = QUERIES) -> None:
    """Raise ValueError if a label cannot be trusted (checked by the tests)."""
    text_of = {passage.key: passage.text for passage in passages}
    if len(text_of) != len(passages):
        raise ValueError("Passage keys must be unique.")
    for passage in passages:
        if passage.language not in LANGUAGES or not passage.text.strip():
            raise ValueError(f"Passage {passage.key!r} has no text or an unknown language.")
    if len({query.query_id for query in queries}) != len(queries):
        raise ValueError("Query IDs must be unique.")
    for query in queries:
        if query.language not in LANGUAGES or not query.text.strip() or not query.relevant:
            raise ValueError(f"Query {query.query_id!r} needs a language, text and evidence.")
        labelled = [(e.passage_key, e.text) for e in (*query.relevant, *query.secondary)]
        if len(set(labelled)) != len(labelled):
            raise ValueError(f"Query {query.query_id!r} labels the same evidence twice.")
        relevant_keys = {e.passage_key for e in query.relevant}
        if relevant_keys & {e.passage_key for e in query.secondary}:
            raise ValueError(f"Query {query.query_id!r}: a passage is relevant AND secondary.")
        for evidence in (*query.relevant, *query.secondary):
            text = text_of.get(evidence.passage_key)
            if text is None or text.count(evidence.text) != 1 or not evidence.text.strip():
                raise ValueError(f"Query {query.query_id!r}: evidence must occur exactly once "
                                 f"in passage {evidence.passage_key!r}.")
        if direction_of(query, passages) not in DIRECTIONS:
            raise ValueError(f"Query {query.query_id!r} has an unknown direction.")
