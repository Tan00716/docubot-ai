Dataset: retrieval_eval_v2 (6fe703fff36429dd468c48e8b4ba666c175d3ef0fd4d73ff13a6f18013dd9afe)

# DocuBot embedding model comparison (Batch 6C)

A development evaluation set (78 chunks, 70 questions; retrieval_eval_v2, SHA-256 6fe703fff36429dd468c48e8b4ba666c175d3ef0fd4d73ff13a6f18013dd9afe), not a general benchmark. Only the embedding model changes; chunking 1200/200, exact search, ranking, top_k and labels are the Batch 6A ones. Deltas are candidate minus intfloat/multilingual-e5-small. One question changes a direction's Recall by 0.2-1.0. Total time 0.5 min.

## Models

| Model | Dimension | Max tokens | Languages | Prefix contract | Pooling | Runtime | Revision |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 384 | 512 | 94 (model card metadata) | query `query: `, passage `passage: ` | mean | FastEmbed custom model, ONNX Runtime CPU, float32 | `614241f622f5` |

Not evaluated:

- **intfloat/multilingual-e5-base**: NOT MEASURED — insufficient safe memory (requires about 2.85 GiB; available 2.27 GiB RAM and 2.86 GiB commit, against the prior 2.28 GiB peak plus 25% reserve).
- **BAAI/bge-m3**: NOT MEASURED — insufficient safe memory (requires about 4.25 GiB; available 2.27 GiB RAM and 2.86 GiB commit, against the prior 3.40 GiB peak plus 25% reserve).

## Overall

| Model | R@1 | R@3 | R@5 | MRR | Δ R@1 | Δ R@3 | Δ R@5 | Δ MRR | Query hits @1/@3/@5 |
|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 0.343 | 0.529 | 0.586 | 0.471 | - | - | - | - | 24/70, 37/70, 41/70 |

## Cross-language (primary metric)

Cross-language = all directions where query and target languages differ (each question counts once).

| Model | Queries | Cross R@1 | Cross R@3 | Cross R@5 | Cross MRR | Δ R@1 | Δ R@3 | Δ R@5 | Δ MRR | Query hits @1/@3/@5 |
|---|---|---|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 36 | 0.083 | 0.250 | 0.306 | 0.212 | - | - | - | - | 3/36, 9/36, 11/36 |

## Language breakdown

| Direction | Model | Queries | R@1 | R@3 | R@5 | MRR | Δ MRR | Query hits @1/@3/@5 |
|---|---|---|---|---|---|---|---|---|
| en->en | intfloat/multilingual-e5-small | 13 | 0.615 | 0.923 | 1.000 | 0.785 | - | 8/13, 12/13, 13/13 |
| zh->zh | intfloat/multilingual-e5-small | 20 | 0.650 | 0.750 | 0.800 | 0.734 | - | 13/20, 15/20, 16/20 |
| zh->en | intfloat/multilingual-e5-small | 12 | 0.000 | 0.083 | 0.083 | 0.051 | - | 0/12, 1/12, 1/12 |
| en->zh | intfloat/multilingual-e5-small | 12 | 0.000 | 0.083 | 0.167 | 0.120 | - | 0/12, 1/12, 2/12 |
| mixed->en | intfloat/multilingual-e5-small | 5 | 0.000 | 0.200 | 0.400 | 0.186 | - | 0/5, 1/5, 2/5 |
| mixed->zh | intfloat/multilingual-e5-small | 5 | 0.600 | 0.800 | 0.800 | 0.729 | - | 3/5, 4/5, 4/5 |
| zh->mixed | intfloat/multilingual-e5-small | 1 | 0.000 | 1.000 | 1.000 | 0.500 | - | 0/1, 1/1, 1/1 |
| en->mixed | intfloat/multilingual-e5-small | 1 | 0.000 | 1.000 | 1.000 | 0.500 | - | 0/1, 1/1, 1/1 |
| mixed->mixed | intfloat/multilingual-e5-small | 1 | 0.000 | 1.000 | 1.000 | 0.500 | - | 0/1, 1/1, 1/1 |

## Where the cross-language change comes from

First relevant rank per cross-language question, in dataset order (zh_en_overlap, zh_en_postgres, zh_en_metrics, zh_en_upload_name, zh_en_vector_bytes, en_zh_injection, en_zh_token_leak, en_zh_hallucination, en_zh_chunk_size, en_zh_cjk_tokens, mixed_en_polling, mixed_en_validation, mixed_en_ann, mixed_zh_batch, mixed_zh_pdf_table, mixed_zh_log_question, zh_mixed_shared_model, en_mixed_async, v2_zh_en_retry, v2_zh_en_rest, v2_zh_en_norm, v2_zh_en_volume, v2_zh_en_python, v2_zh_en_ann, v2_zh_en_timeout, v2_en_zh_put, v2_en_zh_webhook, v2_en_zh_token, v2_en_zh_secret, v2_en_zh_sql, v2_en_zh_overlap, v2_en_zh_retry, v2_mx_en_api, v2_mx_en_timeout, v2_mx_zh_wal, v2_mx_zh_python).

| Model | Queries | Rank 1 | Top 3 | Top 5 | First relevant ranks | Top-5 results in the question's language | Relevant chunk truncated |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 36 | 3 | 9 | 11 | 35 37 - 39 28 6 22 29 35 14 5 22 2 7 1 2 2 2 35 27 47 46 3 49 30 8 4 2 15 25 29 13 17 8 1 1 | 118/180 | 2 |

## Truncation

The same 78 chunks (1200/200) for every model; a chunk is truncated when its model input (prefix included) has more tokens than the model reads.

| Model | Token limit | English | Chinese | Mixed | Total | Longest chunk (tokens) |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 512 | 0/30 (0%) | 5/32 (16%) | 0/16 (0%) | 5/78 (6%) | 650 |

Query outcomes grouped by whether relevant evidence is model-readable:

| Model | Relevant evidence group | Queries | R@1 | R@3 | R@5 | MRR |
|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | not truncated | 60 | 0.350 | 0.550 | 0.600 | 0.479 |
| intfloat/multilingual-e5-small | truncated, evidence inside window | 5 | 0.200 | 0.400 | 0.400 | 0.358 |
| intfloat/multilingual-e5-small | truncated, evidence beyond window | 5 | 0.400 | 0.400 | 0.600 | 0.490 |

## Resources

| Model | Cold download | Cache on disk | Model load | Embed 78 chunks | Query embed + search (mean) | Peak working set | Peak private memory |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 465 MB | 465 MB | 2.4 s | 7.9 s | 41 ms (max 169) | 1.13 GB | 1.48 GB |

Setup (identical for every model): python 3.13.7; platform Windows-11-10.0.26200-SP0; processor Intel64 Family 6 Model 186 Stepping 3, GenuineIntel; logical_cpus 12; onnxruntime 1.30.0; fastembed 0.8.1; threads ONNX Runtime default (not set), identical for every model; embedding batch size 16 (the application default; every passage is its own document). Each model ran in its own fresh process. Timings are single-run measurements on this laptop; treat them as rough indications, not benchmarks.

## Vector and input contract checks (real runs)

| Contract check | intfloat/multilingual-e5-small |
|---|---|
| official_files_match_spec | yes |
| revision_pinned | yes |
| tokenizer_limit | yes |
| deterministic_ranking | yes |
| cpu_only | yes |
| passage_inputs_follow_contract | yes |
| query_inputs_follow_contract | yes |
| no_other_model_inputs | yes |
| dimension | yes |
| finite | yes |
| normalized | yes |
| deterministic_vectors | yes |
| input_hash_is_prefixed_passage | yes |

## Per-query cross-language diagnostics

**intfloat/multilingual-e5-small** (relevant chunks in bold)

| Query | Question | Direction | Expected evidence | First relevant rank | Top 5: language, passage, chunk ID, score |
|---|---|---|---|---|---|
| zh_en_overlap | 为什么相邻的两个文本片段之间要重复一小段内容？ | zh→en | en_overlap: "copies the last part of each piece to the beginning of the following one" | 35 | zh v2_zh_chunk_overlap `9fd10d03…00000` 0.8817<br>zh zh_rag_hallucination `940777ea…00000` 0.8754<br>zh zh_chunk_size_tradeoff `c67c8a96…00000` 0.8708<br>zh zh_tokenization_cjk `45142089…00000` 0.8644<br>zh zh_deterministic_ties `d7e8a5b2…00000` 0.8605 |
| zh_en_postgres | 哪种数据库作为独立的服务进程运行，并且允许很多客户端同时写入？ | zh→en | en_postgres: "many clients can write at the same time" | 37 | zh zh_sqlite_file `b514019d…00000` 0.8825<br>zh v2_zh_sqlite_wal `6cbc5e2a…00000` 0.8810<br>zh v2_zh_sql_param `b19e9883…00000` 0.8775<br>zh zh_sql_injection `37080f1c…00000` 0.8757<br>zh v2_zh_long_faq `d76c894f…00000` 0.8691 |
| zh_en_metrics | 怎么衡量正确的资料在搜索结果中是不是排得足够靠前？ | zh→en | en_eval_metrics: "Mean reciprocal rank looks at the position of the first relevant result" | - | mixed v2_mx_ann `aa9185a7…00000` 0.8747<br>mixed mixed_truncation_512 `7dea553e…00000` 0.8701<br>zh zh_normalize_cosine `9740ffe4…00000` 0.8673<br>zh zh_deterministic_ties `d7e8a5b2…00000` 0.8638<br>zh zh_exact_search `5d1a51af…00000` 0.8616 |
| zh_en_upload_name | 用户上传文件时，怎样防止文件名把文件写到别的目录里去？ | zh→en | en_upload_path: "never uses the client's name on disk" | 39 | zh zh_docker_volume `72b79d65…00000` 0.8782<br>mixed v2_mx_rest_retry `5c6ca52a…00000` 0.8581<br>mixed v2_mx_api_secret `f0059360…00000` 0.8580<br>zh zh_faq `cfdb621c…00000` 0.8555<br>zh zh_sql_injection `37080f1c…00000` 0.8536 |
| zh_en_vector_bytes | 每个向量存进数据库以后大概占多少字节？ | zh→en | en_float32_storage: "1536 bytes" | 28 | zh zh_exact_search `5d1a51af…00000` 0.8853<br>zh zh_stale_embeddings `d2f8c15a…00000` 0.8819<br>zh zh_tokenization_cjk `45142089…00000` 0.8780<br>zh zh_batch_memory `d0e53e19…00000` 0.8767<br>zh zh_chunk_size_tradeoff `c67c8a96…00000` 0.8754 |
| en_zh_injection | How do I stop text typed by a user from being executed as part of a database command? | en→zh | zh_sql_injection: "使用参数化查询" | 6 | en v2_en_tokenizer `2c87c44e…00000` 0.8450<br>en en_sqlite_wal `5bfca4c7…00000` 0.8439<br>en v2_en_sqlite_tx `17875d0f…00000` 0.8356<br>en en_postgres `a12452eb…00000` 0.8319<br>en en_overlap `06ae3aec…00000` 0.8276 |
| en_zh_token_leak | My bot's secret credential ended up in a public repository. Is deleting the commit enough? | en→zh | zh_bot_token: "仅仅删除那次提交是不够的" | 22 | en en_upload_path `3c4a25d3…00000` 0.8369<br>en en_sqlite_wal `5bfca4c7…00000` 0.8349<br>en v2_en_http_retry `cde052e2…00000` 0.8342<br>en en_requirements_pinning `27594a00…00000` 0.8339<br>en v2_en_rest_put `b6307e6f…00000` 0.8331 |
| en_zh_hallucination | Why does a language model sometimes state things that are not in the retrieved material, and how can such claims be checked? | en→zh | zh_rag_hallucination: "这种现象通常称为幻觉" | 29 | en en_e5_prefix `3b55d3dc…00000` 0.8622<br>en v2_en_tokenizer `2c87c44e…00000` 0.8523<br>en en_rag_failure_types `bde857df…00000` 0.8509<br>en en_docker_layers `b9ee9941…00000` 0.8426<br>en en_eval_metrics `d0c5dd03…00000` 0.8424 |
| en_zh_chunk_size | What goes wrong when the pieces a document is split into are very large or very small? | en→zh | zh_chunk_size_tradeoff: "块太大时，一个块里混杂了好几个主题" | 35 | en en_block_split `c0dcd543…00000` 0.8816<br>en en_overlap `06ae3aec…00000` 0.8781<br>en en_rag_failure_types `bde857df…00000` 0.8445<br>en en_e5_prefix `3b55d3dc…00000` 0.8423<br>en v2_en_sqlite_tx `17875d0f…00000` 0.8393 |
| en_zh_cjk_tokens | Why does Chinese text fill up the model's input limit faster than English text of the same length? | en→zh | zh_tokenization_cjk: "中文却可能有六七百个" | 14 | en v2_en_tokenizer `2c87c44e…00000` 0.8714<br>en en_e5_prefix `3b55d3dc…00000` 0.8690<br>en en_overlap `06ae3aec…00000` 0.8550<br>en en_requirements_pinning `27594a00…00000` 0.8462<br>en v2_en_chunk_overlap `528e6bfa…00000` 0.8458 |
| mixed_en_polling | Telegram bot 用 long polling 的时候，是怎么拿到新消息的？ | mixed→en | en_telegram_polling: "repeatedly calls getUpdates" | 5 | mixed v2_mx_telegram_update `a2f27385…00000` 0.9189<br>mixed mixed_webhook_https `93a70385…00000` 0.9187<br>zh v2_zh_telegram_webhook `2920a34b…00000` 0.9115<br>zh zh_bot_token `df4b9ab3…00000` 0.8792<br>**en en_telegram_polling `cc9a86d5…00000` 0.8685** |
| mixed_en_validation | request body 里 top_k 传的是字符串 "5"，FastAPI 会怎么处理？ | mixed→en | en_fastapi_validation: "is not silently converted to the integer 5" | 22 | zh v2_zh_fastapi_422 `5acaa556…00000` 0.8885<br>mixed v2_mx_fastapi_sql `39b0325a…00000` 0.8884<br>mixed mixed_truncation_512 `7dea553e…00000` 0.8789<br>mixed mixed_fastapi_depends `5a287525…00000` 0.8743<br>mixed v2_mx_api_secret `f0059360…00000` 0.8739 |
| mixed_en_ann | FAISS 或者 HNSW 这类 index 为什么有时会漏掉最相似的结果？ | mixed→en | en_ann_index: "a truly closest vector can occasionally be skipped" | 2 | mixed v2_mx_ann `aa9185a7…00000` 0.9196<br>**en en_ann_index `8cf24350…00000` 0.8747**<br>zh v2_zh_embedding_cosine `e56f2a0d…00000` 0.8627<br>zh zh_normalize_cosine `9740ffe4…00000` 0.8607<br>mixed mixed_truncation_512 `7dea553e…00000` 0.8604 |
| mixed_zh_batch | embedding 的 batch size 设得太大会有什么问题？ | mixed→zh | zh_batch_memory: "批次过大可能让内存占用突然升高" | 7 | mixed v2_mx_embedding_score `4fca9531…00000` 0.8884<br>mixed mixed_truncation_512 `7dea553e…00000` 0.8804<br>mixed v2_mx_chunk_token `11929d8e…00000` 0.8746<br>zh zh_chunk_size_tradeoff `c67c8a96…00000` 0.8719<br>zh v2_zh_long_faq `d76c894f…00000` 0.8711 |
| mixed_zh_pdf_table | 从 PDF 里提取出来的 table 顺序全乱了，是什么原因？ | mixed→zh | zh_pdf_extraction: "表格的单元格也常常按坐标顺序连成一行" | 1 | **zh zh_pdf_extraction `4ba0068a…00000` 0.8958**<br>zh zh_faq `cfdb621c…00000` 0.8792<br>zh zh_deterministic_ties `d7e8a5b2…00000` 0.8760<br>zh zh_rag_hallucination `940777ea…00000` 0.8704<br>zh zh_batch_memory `d0e53e19…00000` 0.8666 |
| mixed_zh_log_question | log 里可以直接记录 user 的原始 question 吗？ | mixed→zh | zh_logging_privacy: "而不记录问题原文和返回的文本内容" | 2 | mixed v2_mx_api_secret `f0059360…00000` 0.8597<br>**zh zh_logging_privacy `23d8e32f…00000` 0.8586**<br>zh v2_zh_long_faq `d76c894f…00000` 0.8581<br>zh zh_rag_hallucination `940777ea…00000` 0.8572<br>zh zh_sql_injection `37080f1c…00000` 0.8510 |
| zh_mixed_shared_model | 怎样让所有请求共用同一个已经加载好的模型？ | zh→mixed | mixed_fastapi_depends: "只会真正创建一次 provider" | 2 | mixed mixed_embedding_cache `e133fdb5…00000` 0.8835<br>**mixed mixed_fastapi_depends `5a287525…00000` 0.8818**<br>zh zh_stale_embeddings `d2f8c15a…00000` 0.8771<br>mixed mixed_git_branch `2afeec62…00000` 0.8744<br>mixed mixed_webhook_https `93a70385…00000` 0.8725 |
| en_mixed_async | Should a slow, CPU-heavy model call go into an async endpoint? | en→mixed | mixed_async_endpoints: "整个 event loop 都会被卡住" | 2 | en v2_en_fastapi_async `55b7d194…00000` 0.8661<br>**mixed mixed_async_endpoints `62db43e6…00000` 0.8477**<br>en en_fastapi_validation `a781c89d…00000` 0.8405<br>en en_requirements_pinning `27594a00…00000` 0.8379<br>en v2_en_http_retry `cde052e2…00000` 0.8347 |
| v2_zh_en_retry | 服务返回 429 并给出等待时间时，客户端应该怎么做？ | zh→en | v2_en_http_retry: "respect Retry-After" | 35 | zh v2_zh_http_retry `ff313bde…00000` 0.9159 [HARD NEGATIVE]<br>zh v2_zh_long_faq `d76c894f…00000` 0.8871<br>mixed v2_mx_rest_retry `5c6ca52a…00000` 0.8795<br>mixed mixed_git_branch `2afeec62…00000` 0.8765<br>zh zh_http_422_400 `36aab580…00000` 0.8751 |
| v2_zh_en_rest | 哪种 HTTP 方法通常用于创建子资源，重复发送可能新建多条？ | zh→en | v2_en_rest_put: "POST usually creates a subordinate resource" | 27 | zh v2_zh_rest_idempotent `d980937d…00000` 0.9069 [HARD NEGATIVE]<br>mixed v2_mx_rest_retry `5c6ca52a…00000` 0.8879<br>mixed mixed_webhook_https `93a70385…00000` 0.8673<br>mixed v2_mx_telegram_update `a2f27385…00000` 0.8624<br>mixed mixed_git_branch `2afeec62…00000` 0.8607 |
| v2_zh_en_norm | 单位长度向量的点积和哪个相似度指标相等？ | zh→en | v2_en_embedding_norm: "dot product equals cosine similarity" | 47 | zh v2_zh_embedding_cosine `e56f2a0d…00000` 0.9162 [HARD NEGATIVE]<br>zh zh_normalize_cosine `9740ffe4…00000` 0.9031<br>zh v2_zh_vector_dimensions `eae96b1f…00000` 0.8962<br>mixed v2_mx_embedding_score `4fca9531…00000` 0.8710<br>zh zh_exact_search `5d1a51af…00000` 0.8672 |
| v2_zh_en_volume | 删除 Docker 容器后，容器自身写入层中的文件会怎样？ | zh→en | v2_en_docker_volume: "files written only inside the container disappear with it" | 46 | zh zh_docker_volume `72b79d65…00000` 0.9041<br>mixed v2_mx_docker_volume `629dd232…00000` 0.9020<br>zh v2_zh_docker_image `1a77dc4a…00000` 0.8955 [HARD NEGATIVE]<br>zh v2_zh_long_faq `d76c894f…00000` 0.8687<br>mixed mixed_git_branch `2afeec62…00000` 0.8612 |
| v2_zh_en_python | Python lock file 相比宽泛的 requirements 范围提供了什么？ | zh→en | v2_en_python_lock: "records a resolved set of dependency versions" | 3 | mixed v2_mx_python_env `de1ad43e…00000` 0.8932<br>mixed v2_mx_fastapi_sql `39b0325a…00000` 0.8584<br>**en v2_en_python_lock `f738316d…00000` 0.8517**<br>zh zh_venv `6dfd19a9…00000` 0.8506<br>mixed mixed_truncation_512 `7dea553e…00000` 0.8485 |
| v2_zh_en_ann | 近似向量索引相较精确搜索可能牺牲什么？ | zh→en | v2_en_exact_vs_ann: "may omit a true nearest neighbor" | 49 | zh zh_exact_search `5d1a51af…00000` 0.8902<br>mixed v2_mx_ann `aa9185a7…00000` 0.8889 [HARD NEGATIVE]<br>zh v2_zh_embedding_cosine `e56f2a0d…00000` 0.8824<br>zh zh_stale_embeddings `d2f8c15a…00000` 0.8798<br>zh zh_chunk_size_tradeoff `c67c8a96…00000` 0.8774 |
| v2_zh_en_timeout | HTTP 客户端超时后，为什么不能认定服务器没有处理 POST？ | zh→en | v2_en_http_timeout: "does not prove the server failed to process the request" | 30 | mixed v2_mx_rest_retry `5c6ca52a…00000` 0.8971<br>zh v2_zh_http_retry `ff313bde…00000` 0.8805 [HARD NEGATIVE]<br>zh v2_zh_long_faq `d76c894f…00000` 0.8804<br>zh zh_http_422_400 `36aab580…00000` 0.8802<br>zh zh_rag_hallucination `940777ea…00000` 0.8717 |
| v2_en_zh_put | What makes repeating a PUT operation suitable for retry? | en→zh | v2_zh_rest_idempotent: "相同请求重复执行后，资源状态与执行一次相同" | 8 | en v2_en_rest_put `b6307e6f…00000` 0.8729<br>en v2_en_http_retry `cde052e2…00000` 0.8330<br>en en_rate_limit `f39f5cb6…00000` 0.8234<br>mixed v2_mx_rest_retry `5c6ca52a…00000` 0.8147<br>en v2_en_http_timeout `bca8aabe…00000` 0.8067 |
| v2_en_zh_webhook | What must be reachable for Telegram to deliver updates to a webhook? | en→zh | v2_zh_telegram_webhook: "公网可访问的 HTTPS 地址" | 4 | en en_telegram_polling `cc9a86d5…00000` 0.8729<br>mixed v2_mx_telegram_update `a2f27385…00000` 0.8632 [HARD NEGATIVE]<br>mixed mixed_webhook_https `93a70385…00000` 0.8562<br>**zh v2_zh_telegram_webhook `2920a34b…00000` 0.8501**<br>zh zh_bot_token `df4b9ab3…00000` 0.7796 |
| v2_en_zh_token | Which tool should count actual model tokens for Chinese text? | en→zh | v2_zh_token_limit: "用对应模型的 tokenizer 计算" | 2 | en v2_en_tokenizer `2c87c44e…00000` 0.8746<br>**zh v2_zh_token_limit `c9a697c3…00000` 0.8524**<br>mixed v2_mx_chunk_token `11929d8e…00000` 0.8401<br>mixed mixed_truncation_512 `7dea553e…00000` 0.8336<br>zh zh_tokenization_cjk `45142089…00000` 0.8140 [HARD NEGATIVE] |
| v2_en_zh_secret | If removing a leaked API credential from the source is not enough, what should I do? | en→zh | v2_zh_api_token: "应撤销并轮换" | 15 | en v2_en_api_key `74e12f8b…00000` 0.8463<br>mixed v2_mx_api_secret `f0059360…00000` 0.8379<br>en en_rate_limit `f39f5cb6…00000` 0.8364<br>en en_requirements_pinning `27594a00…00000` 0.8357<br>en v2_en_http_retry `cde052e2…00000` 0.8299 |
| v2_en_zh_sql | How can SQL user input be prevented from becoming executable command text? | en→zh | v2_zh_sql_param: "使用参数占位符并单独传入用户值" | 25 | en en_upload_path `3c4a25d3…00000` 0.8399<br>en en_sqlite_wal `5bfca4c7…00000` 0.8393<br>en v2_en_sqlite_tx `17875d0f…00000` 0.8371 [HARD NEGATIVE]<br>en en_postgres `a12452eb…00000` 0.8287<br>en en_e5_prefix `3b55d3dc…00000` 0.8286 |
| v2_en_zh_overlap | Why copy part of a passage into its neighboring chunk? | en→zh | v2_zh_chunk_overlap: "降低证据句刚好被边界切开的风险" | 29 | en en_overlap `06ae3aec…00000` 0.8706 [HARD NEGATIVE]<br>en v2_en_chunk_overlap `528e6bfa…00000` 0.8704<br>en en_rag_failure_types `bde857df…00000` 0.8379<br>en en_ann_index `8cf24350…00000` 0.8355<br>en en_block_split `c0dcd543…00000` 0.8344 |
| v2_en_zh_retry | How should a client respond to a rate limited API that specifies a wait duration? | en→zh | v2_zh_http_retry: "按其等待" | 13 | en v2_en_http_retry `cde052e2…00000` 0.8675<br>en en_rate_limit `f39f5cb6…00000` 0.8626 [HARD NEGATIVE]<br>en v2_en_fastapi_async `55b7d194…00000` 0.8508<br>en v2_en_http_timeout `bca8aabe…00000` 0.8409<br>en en_fastapi_validation `a781c89d…00000` 0.8326 |
| v2_mx_en_api | API key 放哪里才能避免进入 source control？ | mixed→en | v2_en_api_key: "Keep keys in environment configuration" | 17 | mixed v2_mx_api_secret `f0059360…00000` 0.8992<br>zh v2_zh_api_token `c8a96e35…00000` 0.8656 [HARD NEGATIVE]<br>mixed v2_mx_fastapi_sql `39b0325a…00000` 0.8652<br>mixed v2_mx_python_env `de1ad43e…00000` 0.8545<br>zh v2_zh_long_faq `d76c894f…00000` 0.8517 |
| v2_mx_en_timeout | HTTP request timeout 后 retry POST 为什么有 duplicate 风险？ | mixed→en | v2_en_http_timeout: "can create duplicate effects" | 8 | mixed v2_mx_rest_retry `5c6ca52a…00000` 0.9140 [HARD NEGATIVE]<br>zh v2_zh_rest_idempotent `d980937d…00000` 0.8871<br>zh v2_zh_http_retry `ff313bde…00000` 0.8861<br>zh zh_http_422_400 `36aab580…00000` 0.8751<br>zh v2_zh_long_faq `d76c894f…00000` 0.8728 |
| v2_mx_zh_wal | SQLite WAL 能不能让多个 writer 同时写入？ | mixed→zh | v2_zh_sqlite_wal: "仍然只允许一个写入者" | 1 | **zh v2_zh_sqlite_wal `6cbc5e2a…00000` 0.9340**<br>mixed v2_mx_docker_volume `629dd232…00000` 0.8659<br>zh v2_zh_long_faq `d76c894f…00000` 0.8646<br>zh zh_sqlite_file `b514019d…00000` 0.8608<br>mixed mixed_git_branch `2afeec62…00000` 0.8580 |
| v2_mx_zh_python | Python 里为什么不建议 catch Exception 然后 silent pass？ | mixed→zh | v2_zh_python_exception: "静默忽略会隐藏编程错误" | 1 | **zh v2_zh_python_exception `9839143a…00000` 0.8842**<br>mixed v2_mx_api_secret `f0059360…00000` 0.8621<br>mixed mixed_truncation_512 `7dea553e…00000` 0.8593<br>mixed v2_mx_python_env `de1ad43e…00000` 0.8588<br>mixed v2_mx_chunk_token `11929d8e…00000` 0.8543 |
