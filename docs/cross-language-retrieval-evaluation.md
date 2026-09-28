# Cross-Language Retrieval Evaluation（Batch 6B）

> Batch 6B is historical. Batch 6C dataset expansion, review status, safe-memory
> preflight, and current decision are recorded at the end of this document.

> **这是一个开发评估集上的受控对比，不是通用 benchmark。**
> 所有结论都只针对这 42 个 chunk、33 个问题："On this evaluation set…"。
> 一个问题就能让某个方向的 Recall 变化 0.2–1.0，跨语言组（10 个问题）一个问题 = 0.1。
> 没有做、也不声称任何 statistical significance（统计显著性）。

## 1. 为什么有 Batch 6B

Batch 6A 的评估发现：同语言检索可靠（en→en R@5 = 1.0，zh→zh R@5 = 0.889），
但跨语言检索很弱（zh→en R@5 = 0.0，en→zh R@5 = 0.2）。Batch 6A 的 chunk size 实验证明
**chunking 修不好它**。剩下最直接的问题是：**换一个 multilingual embedding model 能不能修好？**

**Controlled experiment（受控实验）** → 一次只改变一个因素，其他全部固定 → 如果结果变了，
就能说"是这个因素造成的"。本 Batch 只换 embedding model，下面这些都**完全不变**：

| 固定的因素 | 值 |
|---|---|
| 数据集 | Batch 6A 的 42 段文字、33 个问题、证据标签、语言方向（SHA-256 冻结，见 §4） |
| Chunking | 生产默认值 `chunk_size = 1200`，`chunk_overlap = 200`（42 个 chunk，chunk ID 相同） |
| 搜索 | `app.search.service.search()`：exact brute-force，NumPy dot product |
| 排序 | full-precision score DESC，然后 `chunk_id` ASC |
| `top_k` | `MAX_TOP_K`（50，所以全部 42 个 chunk 都会被排名） |
| 指标 | `app/evaluation/metrics.py` 的 Recall@1/3/5、MRR |
| 硬件 / 运行环境 | 同一台笔记本（Intel i5-1334U，12 logical CPUs，16 GB RAM，只用 CPU），Python 3.13.7，onnxruntime 1.30.0，fastembed 0.8.1，ONNX Runtime 默认线程数（不设置），应用默认 embedding batch size 16 |

## 2. 模型与 Model contract（模型契约）

**Model contract** → 一个模型"怎么用才对"的规则：向量维度、最多读多少 token、输入要不要加前缀、
怎么 pooling、要不要 normalize → 例如 E5 必须写成 `query: 你的问题`，而 BGE-M3 不需要任何前缀。
不同模型的 contract 不同，所以 contract 是**每个模型自己的配置**，不是全局的 E5 假设。

所有值都来自模型**官方文件在固定 revision 上的内容**（2026-09-28 检查），运行前还会用下载的文件再核对一次
（`verify_official_files()`）：

| Model | Revision（固定的 commit） | Dimension | Max tokens | Pooling | Query prefix | Passage prefix | 语言 | License |
|---|---|---|---|---|---|---|---|---|
| `intfloat/multilingual-e5-small`（**生产**） | `614241f622f5…` | 384 | 512 | mean | `query: ` | `passage: ` | 94（model card 元数据） | MIT |
| `intfloat/multilingual-e5-base`（候选 A） | `d128750597153bb5987e10b1c3493a34e5a4502a` | 768 | 512 | mean | `query: ` | `passage: ` | 94（model card 元数据） | MIT |
| `BAAI/bge-m3`（候选 B） | `5617a9f61b028005a4858fdac845db406aefb181` | 1024 | 8192 | CLS | 无 | 无 | 100+（model card） | MIT |

来源：`config.json` 的 `hidden_size`、`tokenizer_config.json` 的 `model_max_length`、
`1_Pooling/config.json` 的 `pooling_mode_*`、`modules.json` 里的 `Normalize` 模块。

- **E5 前缀**：model card FAQ——"Do I need to add the prefix "query: " and "passage: "? Yes …
  otherwise you will see a performance degradation."
- **BGE-M3 无前缀**：model card FAQ——BGE-M3 做 dense retrieval 时 "no longer requires adding
  instructions to the queries"；passage 本来就没有 instruction。所以两个前缀都是空字符串。
  测试会在有人给 BGE-M3 加上 `query: ` / `passage: ` 时失败。
- **Normalization**：三个模型的 `modules.json` 都有 `Normalize`，所以检索要求 length-1 向量。
  FastEmbed 自己的 normalization 是关闭的，由 DocuBot 的 `_finish()` 统一做 L2 normalize
  （和生产一样），实际运行时检查每个向量长度 = 1。
- **Runtime**：全部通过**已有的** FastEmbed custom-model 机制（`TextEmbedding.add_custom_model()` +
  `snapshot_download(revision=…)` + `specific_model_path`），ONNX Runtime CPU，float32。
  FastEmbed 0.8.1 没有内置 E5-base 或 dense BGE-M3，所以不能"直接加载"；没有建第二套加载系统，
  没有 PyTorch。BGE-M3 的权重是 ONNX external data（`onnx/model.onnx_data`，2.27 GB），
  所以 spec 多了一个 `external_data_files` 字段。

## 3. 隔离：候选模型不会进入生产

- 候选模型在 [app/evaluation/candidates.py](../app/evaluation/candidates.py) 的 **allowlist** 里，
  **不在** `app.embeddings.config.SUPPORTED_MODELS` 里。应用（API、bot、已存储的向量）无法配置成候选模型：
  `FastEmbedProvider(EmbeddingConfig(model_name="intfloat/multilingual-e5-base"))` 会被拒绝。
- 生产的 `DEFAULT_MODEL_NAME`、`EMBEDDING_VERSION = 2`、384 维 contract、下载的文件列表**都没有改变**
  （有测试逐字段检查）。
- 生产代码只加了向后兼容的小改动：`FastEmbedProvider(config, spec=None)` 的可选 `spec` 参数
  （应用从不传它）、`"cls"` pooling、spec 的 `external_data_files` / `metadata_files` 字段（生产 spec 为空）。
- 每次评估都在**临时文件夹**里建新的 SQLite 知识库，结束后删除；项目的 `storage/metadata` 数据库和已有向量
  不会被打开。不同模型的向量 contract 不同，所以一个模型的向量永远不能被另一个模型搜索（stale，有测试）。
- 结果只输出到终端，不写应用日志，不经过任何 API。

## 4. 评估数据集（完全不变）

- 42 段文字（17 英文、18 中文、7 中英混合）→ 1200/200 下正好 42 个 chunk；33 个问题。
- 方向分布：en→en 5、zh→zh 9、zh→en 5、en→zh 5、mixed→en 3、mixed→zh 3、zh→mixed 1、en→mixed 1、mixed→mixed 1。
- 数据集是静态 Python 代码（`corpus.py` / `golden.py`，与 commit `98fbee9` 逐字节相同）。为了保证可复现，
  它被**冻结**成一个 SHA-256 指纹 `DATASET_SHA256 = 7b8a9e13…c7`：任何段落、问题、标签或方向的改动都会改变指纹，
  对比工具会拒绝运行，测试会失败。
- 每个模型都必须覆盖全部 33 个问题、全部 9 个方向、全部 42 个 chunk（`validate_result()`），
  并且所有模型搜索的 chunk 完全相同（`check_same_chunks()`）。失败的问题不能被悄悄删掉。

## 5. 结果

### Overall

| Model | R@1 | R@3 | R@5 | MRR | Δ R@1 | Δ R@3 | Δ R@5 | Δ MRR |
|---|---|---|---|---|---|---|---|---|
| multilingual-e5-small | 0.455 | 0.606 | 0.667 | 0.555 | – | – | – | – |
| multilingual-e5-base | 0.515 | 0.667 | 0.758 | 0.639 | +0.061 | +0.061 | +0.091 | +0.084 |
| bge-m3 | 未评估（见 §7） | | | | | | | |

E5-small 的数字与 Batch 6A **完全一致**：跨语言弱点在这个数据集上是**可复现的**。

### Cross-language（主要指标：zh→en + en→zh，10 个问题，每个问题算一次）

| Model | Cross R@1 | Cross R@3 | Cross R@5 | Cross MRR | Δ R@1 | Δ R@3 | Δ R@5 | Δ MRR |
|---|---|---|---|---|---|---|---|---|
| multilingual-e5-small | 0.000 | 0.000 | 0.100 | 0.079 | – | – | – | – |
| multilingual-e5-base | 0.100 | 0.200 | 0.400 | 0.254 | +0.100 | +0.200 | +0.300 | +0.175 |

### Language breakdown

| Direction | Queries | E5-small R@1 / R@3 / R@5 / MRR | E5-base R@1 / R@3 / R@5 / MRR | Δ MRR |
|---|---|---|---|---|
| en→en | 5 | 0.800 / 1.000 / 1.000 / 0.900 | 1.000 / 1.000 / 1.000 / 1.000 | +0.100 |
| zh→zh | 9 | 0.778 / 0.778 / 0.889 / 0.810 | 0.667 / 0.778 / 0.889 / 0.766 | −0.044 |
| zh→en | 5 | 0.000 / 0.000 / 0.000 / 0.049 | 0.000 / 0.000 / 0.200 / 0.100 | +0.052 |
| en→zh | 5 | 0.000 / 0.000 / 0.200 / 0.108 | 0.200 / 0.400 / 0.600 / 0.407 | +0.298 |
| mixed→en | 3 | 0.333 / 0.667 / 0.667 / 0.468 | 0.000 / 0.667 / 0.667 / 0.389 | −0.079 |
| mixed→zh | 3 | 0.667 / 1.000 / 1.000 / 0.778 | 0.667 / 1.000 / 1.000 / 0.833 | +0.056 |
| zh→mixed | 1 | 0.000 / 1.000 / 1.000 / 0.500 | 1.000 / 1.000 / 1.000 / 1.000 | +0.500 |
| en→mixed | 1 | 1.000 / 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 / 1.000 | 0.000 |
| mixed→mixed | 1 | 0.000 / 1.000 / 1.000 / 0.500 | 1.000 / 1.000 / 1.000 / 1.000 | +0.500 |

### 跨语言的变化从哪里来

第一个相关 chunk 的排名（数据集顺序）：

| 问题 | 方向 | E5-small | E5-base |
|---|---|---|---|
| zh_en_overlap | zh→en | 20 | 27（变差） |
| zh_en_postgres | zh→en | 19 | 11 |
| zh_en_metrics | zh→en | 28 | 30（变差） |
| zh_en_upload_name | zh→en | 23 | **4** |
| zh_en_vector_bytes | zh→en | 16 | 11 |
| en_zh_injection | en→zh | **4** | **2** |
| en_zh_token_leak | en→zh | 14 | **5** |
| en_zh_hallucination | en→zh | 17 | 6 |
| en_zh_chunk_size | en→zh | 20 | 6 |
| en_zh_cjk_tokens | en→zh | 9 | **1** |

| 来源 | E5-small | E5-base | 解读 |
|---|---|---|---|
| 排第 1 的问题 | 0/10 | 1/10 | 只有 1 个新的 rank-1 成功 |
| 进入前 5 的问题 | 1/10 | 4/10 | 主要变化：**排名整体上升** |
| 前 5 结果中和问题同语言的比例 | 43/50 | 39/50 | 同语言"假阳性"略少，但仍然占多数 |
| 相关 chunk 被截断的跨语言问题 | 1 | 1 | **不是**截断减少带来的（两个模型 token 上限都是 512） |

- **en→zh 5/5 都变好**（中位排名 14 → 5）；**zh→en 只有 3/5 变好、2 个变差**，中位排名 20 → 11，仍然只有 1/5 进入前 5。
- 同语言方向没有一致的提升：en→en 变好，zh→zh 和 mixed→en 各掉了一个 rank-1。
- 每个方向 1–9 个问题，所以这些都是**观察到的变化**，证据不足以推广到其他数据。

### Truncation（同样的 42 个 chunk）

| Model | Token limit | English | Chinese | Mixed | Total | 最长 chunk（tokens） |
|---|---|---|---|---|---|---|
| multilingual-e5-small | 512 | 0/17（0%） | 4/18（22%） | 0/7（0%） | 4/42（10%） | 650 |
| multilingual-e5-base | 512 | 0/17（0%） | 4/18（22%） | 0/7（0%） | 4/42（10%） | 650 |
| bge-m3 | 8192 | 未测量（未加载） | | | | |

E5-base 和 E5-small 用的是**同一个 XLM-R tokenizer、同一个 512 上限**，所以截断完全相同：
**E5-base 不能缓解中文截断问题**。BGE-M3 的 8192 上限在纸面上会让这 42 个 chunk 都不被截断（最长 650 tokens，
按 E5 tokenizer 计算），但因为没有加载，这一点**没有被实际测量**。

## 6. 资源（实测，同一台笔记本，每个模型在独立的新进程里）

| Model | Cold download | Cache 大小 | 加载模型 | Embed 42 chunks | Query embed + search（平均） | Peak working set | Peak private memory |
|---|---|---|---|---|---|---|---|
| multilingual-e5-small | 465 MB | 465 MB | 2–4 s | 4.7–5.0 s | 35–42 ms | 1.22 GB | 1.67 GB |
| multilingual-e5-base | 1.05 GB | 1.05 GB | 5–8 s | 9.9–10.5 s | 76–89 ms | 1.85 GB | 2.28 GB |
| bge-m3 | 2.13 GB | 2.13 GB（已下载） | 未测量 | 未测量 | 未测量 | 未测量（估计约 3.4 GB） | |

- 范围 = 同一天的两次独立运行（检索结果完全相同，只有计时不同）。Windows 上计时波动明显，只当作大致范围。
- 内存来自操作系统（Win32 `GetProcessMemoryInfo`），因为 ONNX 权重在 native 内存里，Python 自己看不到。
- 向量存储：E5-small 384 × 4 = 1536 bytes/chunk，E5-base 768 × 4 = 3072 bytes（2 倍），BGE-M3 4096 bytes。
- 全部 CPU-only（`CPUExecutionProvider`，运行时检查）。
- 下载的只有 tokenizer / config JSON 和 ONNX 文件；没有 `.py`、`.bin`、`.pt`、pickle，模型仓库里的代码不会被执行。

## 7. 为什么 BGE-M3 被跳过

> **Candidate B skipped due to measured resource / compatibility constraints.**

- 对比工具在运行 BGE-M3 之前先做一个**基于实测的资源检查**：用本次实际测到的两个 E5 点
  （下载大小 → peak private memory：0.45 GB → 1.67 GB，1.05 GB → 2.28 GB）画一条直线，外推到 BGE-M3
  的 2.13 GB，估计 peak 约 **3.4 GB**。
- 运行时这台 16 GB 笔记本上**可用 RAM 只有 2.3 GB**（其他程序占用了大部分内存；commit headroom 5.3 GB）。
  估计值 > 可用 RAM，意味着 Windows 会把其他程序换页到磁盘，计时失去意义，还可能影响用户正在用的程序，
  所以按规则**不强行运行**。
- 模型文件已经按固定 revision 下载并通过官方文件核对（1024 维、8192 tokens、CLS pooling、Normalize）。
  但 FastEmbed 是否能正确加载它的 ONNX external data、输出是否为 `last_hidden_state`，**没有被验证**。
- 可用内存足够时（例如关闭其他程序，可用 RAM ≥ 约 4 GB），直接重新运行同一个命令，BGE-M3 会自动参与对比。

## 8. 回答 Batch 6B 的问题

| 问题 | 在这个评估集上的回答 |
|---|---|
| 1. E5-small 的跨语言弱点可复现吗？ | **是**。两次独立运行、与 Batch 6A 完全相同（Cross R@5 0.10，MRR 0.079）。 |
| 2. E5-base 有改善吗？ | **有，但不够**。Cross R@5 0.10 → 0.40，MRR 0.079 → 0.254；主要来自 en→zh；zh→en 仍然很弱（R@5 0.2）；6/10 个跨语言问题仍不在前 5。 |
| 3. 另一个 multilingual 模型能更好吗？ | **未知**。BGE-M3 因实测资源限制被跳过。 |
| 4. Latency / memory tradeoff？ | E5-base：下载 2.3 倍，embedding 约 2 倍慢，每次查询约 2 倍慢（~80 ms），peak 内存 +0.6 GB，向量 2 倍大；截断没有改善。 |
| 5. 改善值得迁移架构吗？ | **目前不值得**。提升基于 10 个问题中的几个，zh→en 基本没解决，同语言方向有得有失，而迁移需要重新生成全部 embeddings 和 contract 版本。 |
| 6. 结果大到足以改变生产决定吗？ | **不够**。它说明"模型大小对 en→zh 有帮助"，但不是清楚的迁移证据。 |

## 9. Decision record：生产 embedding model

| | |
|---|---|
| **决定** | **MORE-DATA**：暂时 **KEEP `multilingual-e5-small`**，先收集更多评估数据（选项 4）。同时记录：**仅换成 E5-base 并不能充分解决跨语言问题**（选项 5 的观察）。 |
| **日期** | 2026-09-28（Batch 6B） |
| **依据** | ① E5-base 的跨语言提升是真实观察到的（Cross R@5 +0.30，MRR +0.175），但只基于 10 个问题，其中 zh→en 5 个问题里 2 个变差；② E5-base 不改变截断（同一 tokenizer、同一 512 上限）；③ 同语言方向有得有失（zh→zh −0.044，mixed→en −0.079 MRR）；④ 成本约 2 倍（延迟、内存、存储、下载）；⑤ 最有希望同时解决跨语言和中文截断的 BGE-M3 没能在这台机器上实测。 |
| **不做的事** | 不迁移生产模型、不重新生成 embeddings、不改 `EMBEDDING_VERSION`、不改 chunking / `/search`。 |
| **如果以后迁移** | E5-base 或 BGE-M3 看起来有希望；**生产迁移必须是单独的 Batch**（schema / embedding 重新生成 / 版本迁移不应和评估混在一起）。 |
| **重新评估的条件** | 更多跨语言问题（尤其 zh→en）和更长的中文文档；在可用 RAM 足够时完成 BGE-M3 的测量。 |

## 10. 如何运行

```powershell
# 1) 一次性、显式下载候选模型（唯一需要网络的一步；只接受 allowlist 名字，固定 revision）
.venv\Scripts\python.exe -m app.evaluation.download_candidates e5-base
.venv\Scripts\python.exe -m app.evaluation.download_candidates bge-m3   # 可选，2.13 GB

# 2) 离线对比（HF_HUB_OFFLINE=1 自动设置；没缓存的模型会被报告为 skipped，不会下载）
.venv\Scripts\python.exe -m app.evaluation.compare_models
.venv\Scripts\python.exe -m app.evaluation.compare_models --models e5-small,e5-base
```

**Model cache 怎么区分**：所有模型都在 `storage/model_cache/`（已 git-ignore，永远不提交），
`huggingface_hub` 按"仓库 + commit"分文件夹：`models--intfloat--multilingual-e5-base/snapshots/d1287505…/`。
不同模型、同一模型的不同 revision 都不会混在一起；对比工具只接受 `snapshots/<固定 revision>/` 里的文件。

代码：

| 文件 | 作用 |
|---|---|
| [candidates.py](../app/evaluation/candidates.py) | 候选模型 allowlist、model contract、官方文件核对 |
| [comparison.py](../app/evaluation/comparison.py) | 冻结的数据集指纹、结果记录、跨语言指标、delta、截断、完整性检查（纯函数） |
| [compare_models.py](../app/evaluation/compare_models.py) | 离线 CLI：每个模型一个子进程、真实输入记录、向量 contract 检查、资源检查 |
| [comparison_report.py](../app/evaluation/comparison_report.py) | Markdown 报告（包括每个跨语言问题的 top-5 诊断） |
| [download_candidates.py](../app/evaluation/download_candidates.py) | 显式下载（复用生产的 `download_model_files`） |
| [resources.py](../app/evaluation/resources.py) | 内存与磁盘测量（Win32 API，无额外依赖） |

每次真实运行都会检查并报告：官方文件与 spec 一致、revision 固定、tokenizer 上限、模型实际收到的输入
正好是 `passage_prefix + chunk` 和 `query_prefix + 问题`（没有交换、缺失或多余的前缀）、维度、有限值、
length-1、同一文本两次得到相同向量、两次评估排名完全相同、只用 CPU。两个 E5 模型全部通过。

## 11. Limitations（局限）

- 33 个问题、10 个跨语言问题：一个问题 = 跨语言 Recall 0.1。所有差异都只是"在这个数据集上观察到的"。
- 问题和标签由同一人（AI 辅助）编写，没有独立标注。语料是短段落，每段一个文档。
- 每个语料主题都有同语言的相近段落，这会放大"同语言优先"的效果。
- BGE-M3 没有被实测（资源限制）；只比较了一个候选。
- 只比较 dense retrieval；没有 reranking、hybrid search（本 Batch 明确不做）。
- 计时来自一台负载较高的 Windows 笔记本，只是大致范围。

---

# Batch 6C — Retrieval Evaluation Expansion & BGE-M3 Measurement

Batch 6C adds a frozen larger dataset and safe resource preflight. It does not
change the production embedding model, stored vectors, search route, or chunking.
The Batch 6A and 6B results above remain historical records for the original data.

## Dataset Versions

| Version | Passages | Queries | Chunks at 1200/200 | Status |
|---|---:|---:|---:|---|
| `retrieval_eval_v1` | 42 | 33 | 42 | Immutable Batch 6A/6B baseline |
| `retrieval_eval_v2` | 78 | 70 | 78 | Batch 6C expansion; synthetic/project-authored content |

The artifact SHA-256 is over canonical UTF-8 JSON, including version, passages,
query text, language tags, target language, supporting evidence spans, secondary
labels, and hard-negative assignments.

| Version | SHA-256 |
|---|---|
| `retrieval_eval_v1` | `c5e2cc8dc6198facfd9c2e1c8ca2cd2337a2d86be659b373318d6bde57c3637a` |
| `retrieval_eval_v2` | `6fe703fff36429dd468c48e8b4ba666c175d3ef0fd4d73ff13a6f18013dd9afe` |

The v1 execution fingerprint used by Batch 6B remains `7b8a9e13b131e4042cc13f4d77aeb66a876a2f947900f066a97b97cb8ee921c7`.
Dataset JSON files are committed under [app/evaluation/datasets](../app/evaluation/datasets/).
Evaluation verifies both the artifact digest and the source records before it runs;
existing artifacts cannot be overwritten by the maintainer helper unless `force=True`
is explicitly supplied.

The v2 corpus distribution is 30 English, 32 Chinese, and 16 mixed passages. Its
queries are: en→en 13, zh→zh 20, zh→en 12, en→zh 12, mixed→en 5, mixed→zh 5,
zh→mixed 1, en→mixed 1, and mixed→mixed 1. This yields 36 cross-language queries
across all unlike-language directions, including 12 zh→en and 12 en→zh.

At least one labeled hard negative is listed for each of the 37 newly added queries.
These are plausible, nearby but incorrect passages. They remain non-relevant for
the main binary Recall@K and MRR metrics; they are marked in per-query diagnostics
when they appear in a model's top five.

## Independent Relevance Review

A blind review pack for the 37 newly added queries is available at
[retrieval_eval_v2_blind_review.json](retrieval_eval_v2_blind_review.json). It
contains only the query, candidate source passages, language tags, and review
criteria. Query and candidate IDs are opaque and candidates are reordered; the pack contains
no model names, scores, rankings, or relevance labels. The rubric is:

- `RELEVANT`: directly supports the answer.
- `PARTIALLY_RELEVANT`: related context but insufficient to answer.
- `NOT_RELEVANT`: does not support the answer.

Only `RELEVANT` counts as positive for the main binary Recall@K and MRR metrics.
`PARTIALLY_RELEVANT` is retained for review notes and counts as not relevant in
those primary metrics.

The review workflow requires one judgment per candidate; a question explicitly
marked ambiguous is removed from the accepted review set, and a non-ambiguous
question with no relevant evidence is rejected rather than kept to inflate the set.

The second pass reviewed all 37 new queries against the question and passages,
without consulting model output. No relevance labels changed and no query was
removed as ambiguous. The authoring assistant also performed this review, so it is
a blinded self-review rather than independent human corroboration. There was one
reviewer; agreement and formal inter-rater reliability are not measurable. The
labels should remain provisional until a second person reviews the pack.

## Expanded Retrieval Results

E5-small was measured on v2. E5-base and BGE-M3 were skipped by the memory
preflight, not marked as failed models.

| Model | R@1 | R@3 | R@5 | MRR | Query hits @1 / @3 / @5 |
|---|---:|---:|---:|---:|---|
| E5-small | 0.343 | 0.529 | 0.586 | 0.471 | 24/70, 37/70, 41/70 |
| E5-base | — | — | — | — | NOT MEASURED — insufficient safe memory |
| BGE-M3 | — | — | — | — | NOT MEASURED — insufficient safe memory |

The result rows, per-direction counts, truncation groups, checks, resource data,
and complete cross-language top-five diagnostics are in the
[expanded comparison report](retrieval_eval_v2_model_comparison.md).

The v2 runner reports successful query counts beside each metric when models are
measured. It uses the same passages, labels, 1200/200 chunking, exact SQLite/NumPy
search, ranking, top-k, and evaluation functions for all candidates; only the
model-required input prefixes vary.

## Cross-Language Results

| Group | Queries | R@1 | R@3 | R@5 | MRR | Hits @1 / @3 / @5 |
|---|---:|---:|---:|---:|---:|---|
| All unlike-language directions | 36 | 0.083 | 0.250 | 0.306 | 0.212 | 3/36, 9/36, 11/36 |
| zh→en | 12 | 0.000 | 0.083 | 0.083 | 0.051 | 0/12, 1/12, 1/12 |
| en→zh | 12 | 0.000 | 0.083 | 0.167 | 0.120 | 0/12, 1/12, 2/12 |

The zh→en weakness remains visible with 12 queries, and en→zh also has no rank-1
hits. E5-base was not measured on this expanded set, so its Batch 6B improvement
cannot be confirmed here. The all-cross-language aggregate includes mixed and
mixed-target directions and should not be compared as if it were the old
zh→en/en→zh-only 10-query aggregate. The detailed per-query ranks and top-five
languages/scores are in the linked report.

## Truncation Results

Token counts below use the cached tokenizer JSON from the exact production
E5-small revision and include the `passage: ` prefix and special tokens. This is a
tokenizer-only analysis; no embedding model was loaded.

| Language | Truncated chunks | Total chunks |
|---|---:|---:|
| English | 0 | 30 |
| Chinese | 5 | 32 |
| Mixed | 0 | 16 |
| Total | 5 | 78 |

Relevant-evidence query groups are 60 not truncated, 5 truncated with evidence
inside the readable prefix, and 5 with evidence beyond it. At E5-small's measured
rankings, the group metrics were:

| Evidence group | Queries | R@1 | R@3 | R@5 | MRR | Query hits @1 / @3 / @5 |
|---|---:|---:|---:|---:|---:|---|
| Not truncated | 60 | 0.350 | 0.550 | 0.600 | 0.479 | 21/60, 33/60, 36/60 |
| Truncated, evidence inside window | 5 | 0.200 | 0.400 | 0.400 | 0.358 | 1/5, 2/5, 2/5 |
| Truncated, evidence beyond window | 5 | 0.400 | 0.400 | 0.600 | 0.490 | 2/5, 2/5, 3/5 |

These small groups do not establish that truncation improves retrieval. The
outside-window group still missed relevant evidence for two of five queries at
top five, while topic similarity sometimes ranked a truncated chunk. The five outside cases
are four historical long Chinese answers (two other Chinese passages and two
entries in the original FAQ) plus the new late FAQ answer about Docker volumes. The long v2
FAQ is one 881-character chunk: 538 model-input tokens; evidence ends at token 35
(beginning), 310 (middle), and 517 (end). Thus the last evidence sentence falls
outside the 512-token production E5 window. E5-small's measured ranking for the
query targeting that final answer is shown in the linked report; its result is
one observation and does not establish the general retrieval impact of truncation.

## BGE-M3 Measurement

**NOT MEASURED — insufficient safe memory.** The pinned BGE-M3 cache is present
and occupies approximately 2.13 GiB on disk. After E5-small completed, available
physical memory was about 2.27 GiB and available commit about 2.86 GiB, below the
4.25 GiB safe preflight requirement based on the prior ~3.4 GiB peak plus reserve.
The process was not started and no BGE-M3 accuracy, latency, memory peak, or
truncation result is claimed. The cache is Git-ignored and was left intact.

## Resource Comparison

E5-small measured 2.4 s model load, 7.9 s to embed 78 chunks, 41 ms mean query
embedding plus search (169 ms maximum), 1.13 GiB peak working set, and 1.48 GiB
peak private memory. Its cached model files occupied 465 MiB. E5-base and BGE-M3
were skipped before model load: available RAM/commit after the E5-small run was
2.27/2.86 GiB; their safe requirements were 2.85/4.25 GiB. The measurement setup
was Python 3.13.7, FastEmbed 0.8.1, ONNX Runtime 1.30.0, CPU execution, and batch
size 16 on the same 16 GB Intel i5-1334U Windows laptop. These are single-run
values, not ranges. Earlier Batch 6A/6B measurements remain historical and are
not represented as v2 results. The evaluator checks physical and commit headroom
before each child, uses a 25% reserve over the prior peak, runs offline, and
isolates each model in a fresh process. `--skip-bge` and `--only-bge` are supported.

## Decision

**MORE-DATA.** Keep production on E5-small. The new corpus raises cross-language
coverage from the historical 10 cases to 36 and adds realistic long Chinese text.
E5-small's expanded-set measurements still show weak cross-language retrieval;
E5-base and BGE-M3 were not measured on v2, and the labels do not yet have
independent human review. There is no evidence in this batch to justify migration
or to conclude whether the E5-base improvement persists. Rerun the v2 comparison
on a host with safe memory headroom and have a second reviewer adjudicate the blind
pack. No production migration, embedding regeneration, or chunking change occurred.

## Tests and Negative Testing

| Suite | Count |
|---|---:|
| Existing tests before Batch 6C | 535 (517 passed, 18 opt-in skips) |
| New Batch 6C tests | 39 |
| Current total | 574 (556 passed, 18 opt-in skips) |

The 15 requested negative scenarios were caught by temporary artifact/result
mutations, resource preflight, or existing isolation and validation tests (15/15):
gold query/passage/label/checksum changes; missing directions or cross-language
queries; removed hard negatives; explicitly ambiguous review entries; production
candidate mutation; forced BGE-M3 below its memory reserve; offline cache misses;
changed prefix contract; altered relevance; changed ranking fingerprint; and
different chunks across model results. Mutations used disposable test data/results;
no repository source mutation was left behind. The ranking mutation was simulated
at the evaluated-result fingerprint level.

The new tests specifically cover dataset immutability and digests, language/query
counts, evidence windows, review disposition, BGE selection and memory guard, and
determinism. The 18 skips are existing opt-in real-model smoke/evaluation tests;
the separate Batch 6C comparison did load E5-small after its safe-memory preflight.

Run the versioned evaluation offline:

```powershell
.venv\Scripts\python.exe -m app.evaluation.compare_models --dataset-version retrieval_eval_v2
.venv\Scripts\python.exe -m app.evaluation.compare_models --dataset-version retrieval_eval_v2 --skip-bge
.venv\Scripts\python.exe -m app.evaluation.compare_models --dataset-version retrieval_eval_v2 --only-bge
```
