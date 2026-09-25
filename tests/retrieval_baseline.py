"""A tiny, hand-labelled retrieval baseline (development only).

It answers one narrow question: "does the current embedding model + exact
search put the chunk we know is right near the top?" It is NOT a benchmark
against other models, NOT a production quality claim, and NOT the future
evaluation framework. It exists so later batches (chunking, reranking,
another model) can be compared with a fixed starting point.

Every passage becomes its own one-chunk document. Some passages are
deliberately close to others (FastAPI vs. Flask, SQLite vs. PostgreSQL,
Telegram vs. email) so the task is not trivially easy. Queries are English,
Chinese, and cross-language (Chinese question, English passage).
"""

from collections.abc import Mapping, Sequence

# key -> passage text (English and Chinese)
PASSAGES = {
    "fastapi": "FastAPI is a modern Python web framework for building HTTP APIs. It uses type "
               "hints to validate requests and generates OpenAPI documentation automatically.",
    "flask": "Flask is a lightweight Python micro-framework for web applications. It gives you "
             "routing and templates and leaves most other choices to extensions.",
    "sqlite": "SQLite is an embedded relational database. The whole database is a single file, "
              "and it runs inside the application process without a separate server.",
    "postgres": "PostgreSQL is a client-server relational database. A database server process "
                "accepts network connections from many clients at the same time.",
    "telegram": "Telegram bots are programs that receive messages from users through the "
                "Telegram Bot API and send replies back, for example with long polling.",
    "email": "SMTP is the protocol used to send email between mail servers. Mail clients use "
             "IMAP or POP3 to download messages from a mailbox.",
    "embeddings": "Text embeddings represent a piece of text as a list of numbers, a vector. "
                  "Texts with similar meaning get vectors that point in similar directions.",
    "chunking": "Chunking splits a long document into smaller overlapping pieces of text so "
                "that each piece can be embedded and retrieved on its own.",
    "docker": "Docker packages an application and its dependencies into a container image, so "
              "it runs the same way on every machine.",
    "git": "Git is a distributed version control system. Every commit records a snapshot of "
           "the project, and branches let developers work in parallel.",
    "zh_weather": "新加坡位于赤道附近，属于热带雨林气候，全年高温多雨，没有明显的四季变化。",
    "zh_rice": "煮米饭之前先把大米淘洗干净，米和水的比例大约是一比一点二，大火煮开后转小火焖十五分钟。",
    "zh_normalize": "向量归一化就是把向量缩放到长度为一。归一化之后，两个向量的余弦相似度就等于它们的点积。",
    "zh_python_venv": "Python 虚拟环境为每个项目提供独立的软件包目录，这样不同项目的依赖版本就不会互相冲突。",
    "zh_http_404": "HTTP 状态码 404 表示服务器找不到请求的资源，而 500 表示服务器内部发生了错误。",
    "zh_tea": "绿茶的冲泡水温一般在八十度左右，水温太高会让茶叶变苦，冲泡时间大约两到三分钟。",
}

# query -> the passage keys that answer it (manually labelled)
QUERIES = {
    "Which Python framework builds APIs and creates OpenAPI docs?": {"fastapi"},
    "Which database stores everything in one file without a server?": {"sqlite"},
    "How does a chatbot receive and answer user messages?": {"telegram"},
    "How is text turned into numbers for semantic search?": {"embeddings"},
    "新加坡的天气怎么样？": {"zh_weather"},
    "为什么归一化以后可以用点积计算余弦相似度？": {"zh_normalize"},
    "如何让不同项目的 Python 依赖互不影响？": {"zh_python_venv"},
    "为什么要把长文档切成小块？": {"chunking"},  # cross-language: Chinese question, English passage
}


def recall_at_k(
    ranked: Mapping[str, Sequence[str]], relevant: Mapping[str, set[str]], k: int
) -> float:
    """Average over queries of: (relevant items in the top k) / (all relevant items).

    ranked[query] is the search result order (best first); relevant[query]
    the labelled answers. With one relevant item per query this is the share
    of queries whose answer appears in the top k.
    """
    if not relevant:
        raise ValueError("At least one query is needed.")
    total = 0.0
    for query, answers in relevant.items():
        if not answers:
            raise ValueError(f"Query {query!r} has no relevant items.")
        if query not in ranked:
            raise ValueError(f"Query {query!r} has no ranking.")
        found = answers.intersection(ranked[query][:k])
        total += len(found) / len(answers)
    return total / len(relevant)
