"""The evaluation corpus: hand-written passages about DocuBot's own domain.

Each passage becomes ONE uploaded .txt document; the real chunker then
splits it (at 1200/200 every passage fits into one chunk, smaller chunk
sizes split the long ones). Relevance is NOT attached to chunk IDs but to
evidence sentences (see golden.py), so the labels stay valid for every
chunk size.

Languages:
    en    - English
    zh    - Chinese (technical words may appear in Latin letters, e.g. SQLite)
    mixed - Chinese sentences with many English technical terms, the way
            Chinese developers often really write

The corpus deliberately contains near neighbours that share vocabulary but
answer different questions (SQLite WAL vs SQLite file vs PostgreSQL, 400 vs
422 vs 404, long polling vs webhook, exact search vs ANN index, ...), short,
medium and long passages, and long Chinese passages whose later sentences
lie beyond the model's 512-token window.

The texts are written for this evaluation: no user data, no secrets.
"""

from dataclasses import dataclass
from typing import Literal

Language = Literal["en", "zh", "mixed"]
LANGUAGES: tuple[Language, ...] = ("en", "zh", "mixed")


@dataclass(frozen=True)
class Passage:
    key: str  # stable name used by the golden labels
    language: Language
    text: str


def _en(key: str, text: str) -> Passage:
    return Passage(key, "en", text)


def _zh(key: str, text: str) -> Passage:
    return Passage(key, "zh", text)


def _mixed(key: str, text: str) -> Passage:
    return Passage(key, "mixed", text)


PASSAGES: tuple[Passage, ...] = (
    # --- English ---------------------------------------------------------------
    _en("en_overlap",
        "When a long text is cut into pieces for retrieval, a sentence that sits exactly on a "
        "cut can end up split in two, with half of its meaning in each piece. Neither half may "
        "then match a question about it. To reduce this risk, the splitter copies the last part "
        "of each piece to the beginning of the following one. The copied region is usually a "
        "small fraction of the piece length, for example 200 characters for 1200-character "
        "pieces. The price is some duplicated storage and a few more vectors to compute."),
    _en("en_block_split",
        "DocuBot's splitter respects the structure that the parser found: PDF pages, Word "
        "paragraphs and blank-line separated blocks in plain text. Small blocks are packed "
        "together until adding the next one would exceed the size limit, and a block is only "
        "cut when it is larger than the limit on its own. Keeping paragraphs intact means that "
        "a retrieved piece usually reads like a coherent part of the original document rather "
        "than a random window of characters."),
    _en("en_sqlite_wal",
        "By default SQLite uses a rollback journal: while one connection writes, readers can be "
        "blocked until the transaction finishes. Switching the database to write-ahead logging "
        "(journal_mode=WAL) lets readers keep reading the last committed state while a single "
        "writer appends its changes to a separate -wal file. There is still only one writer at "
        "a time, so a busy web server that writes constantly may see 'database is locked' "
        "errors unless it retries or sets a busy timeout."),
    _en("en_postgres",
        "PostgreSQL runs as a separate server process. Applications connect to it over the "
        "network, authenticate with a user name and password, and many clients can write at the "
        "same time thanks to row-level locking."),
    _en("en_sqlite_migration",
        "When a new column is added to an existing SQLite table, older database files on users' "
        "machines do not have it yet. A small startup migration can read PRAGMA table_info and "
        "run ALTER TABLE ... ADD COLUMN only for the columns that are missing, so new and old "
        "database files end up with the same schema without losing any rows."),
    _en("en_fastapi_validation",
        "FastAPI validates every request body against a Pydantic model before the endpoint "
        "function runs. With strict fields, the JSON string \"5\" is not silently converted to "
        "the integer 5; the request is refused with status 422 and a list of the fields that "
        "failed, so the handler never sees malformed input. Unknown fields can be forbidden with "
        "extra=\"forbid\"."),
    _en("en_flask",
        "Flask is a small Python web framework. It provides routing and templates, and leaves "
        "choices such as database access and input validation to extensions."),
    _en("en_telegram_polling",
        "A Telegram bot can learn about new messages in two ways. With long polling, the bot "
        "itself repeatedly calls getUpdates; each call waits on Telegram's side until an update "
        "arrives or a timeout passes, and the offset parameter tells Telegram which updates were "
        "already handled. Nothing has to be reachable from the internet, which makes this the "
        "easiest option on a laptop."),
    _en("en_docker_layers",
        "Every instruction in a Dockerfile produces a cached layer, and a layer is rebuilt only "
        "when its inputs change. If the whole source tree is copied before pip install runs, "
        "editing any single file invalidates the cache and every package is downloaded again. "
        "Copying requirements.txt first, installing, and copying the rest of the code afterwards "
        "keeps the slow dependency layer cached between code changes."),
    _en("en_e5_prefix",
        "The E5 family of embedding models was trained with two input markers: questions start "
        "with 'query: ' and documents start with 'passage: '. The markers tell the model which "
        "side of the search it is looking at, because a short question and the paragraph that "
        "answers it are worded very differently. Leaving the markers out, or giving a question "
        "the document marker, still produces vectors, but they are measurably worse at finding "
        "the right paragraph."),
    _en("en_ann_index",
        "Approximate nearest neighbour indexes such as HNSW graphs or FAISS IVF lists avoid "
        "comparing the query with every stored vector. They visit only a promising part of the "
        "collection, which makes search on millions of vectors fast, but a truly closest vector "
        "can occasionally be skipped. Parameters like efSearch or nprobe trade speed against "
        "that recall loss."),
    _en("en_rag_failure_types",
        "When a document assistant gives a wrong answer, first find out which stage failed. "
        "Look at the passages that were retrieved for the question. If none of them contains "
        "the needed fact, the problem is retrieval: the embedding, the chunking or the ranking "
        "did not surface the evidence, and no prompt engineering can fix that. If the right "
        "passage was retrieved but the answer contradicts it or adds facts that are not there, "
        "the problem is generation. A third kind of failure sits between them: the evidence was "
        "found but dropped while building the context, for example because too many passages "
        "were cut to fit the prompt. Keeping the retrieved passages and their scores in the "
        "logs of a test run makes this diagnosis possible, and it is the reason retrieval is "
        "usually evaluated on its own before any answer is generated. A development evaluation "
        "set with known relevant passages turns vague impressions like 'search feels worse' "
        "into numbers that can be compared before and after a change."),
    _en("en_rate_limit",
        "When a client sends requests faster than an API allows, the server answers with status "
        "429 Too Many Requests. A Retry-After header can state how many seconds the client "
        "should wait before trying again."),
    _en("en_requirements_pinning",
        "Writing exact versions such as fastapi==0.141.1 into requirements.txt makes an "
        "installation reproducible: a fresh machine gets the same libraries that the tests "
        "passed with. Without pins, a new release of a dependency can change behaviour overnight "
        "even though no line of the project changed."),
    _en("en_eval_metrics",
        "Two numbers are common for judging a search system on a set of questions with known "
        "answers. Recall at K asks whether the relevant passages appear among the first K "
        "results; with a single relevant passage it is simply the share of questions whose "
        "answer is in the top K. It ignores the exact position inside those K results. Mean "
        "reciprocal rank looks at the position of the first relevant result: one for rank one, "
        "one half for rank two, one quarter for rank four, and zero when it is missing. "
        "Averaged over all questions it rewards systems that put the evidence at the very top, "
        "which matters when only the first few passages are passed on to a language model. "
        "Neither number says whether an answer is well written; they only measure whether the "
        "evidence was found and how high it was placed."),
    _en("en_upload_path",
        "A file name sent by a browser is untrusted input. A name like ../../app/config.py or an "
        "absolute Windows path could make the server write outside its upload folder. DocuBot "
        "therefore never uses the client's name on disk: it stores every upload under a newly "
        "generated random identifier plus an allowed extension, and keeps the original name "
        "only as metadata."),
    _en("en_float32_storage",
        "Each stored vector has 384 numbers saved as 32-bit floats in little-endian order, so "
        "one embedding takes exactly 1536 bytes in the database."),
    # --- Chinese ---------------------------------------------------------------
    _zh("zh_sqlite_file",
        "SQLite 把整个数据库保存在一个普通文件里，不需要单独安装或启动数据库服务器。应用程序通过函数库直接读写"
        "这个文件，备份时复制文件即可，因此很适合本地开发和小型项目。"),
    _zh("zh_sql_injection",
        "如果把用户输入直接拼接进 SQL 字符串，攻击者就可以输入一段精心构造的文本，改变整条语句的含义，例如绕过"
        "登录检查或者删除数据表。正确的做法是使用参数化查询：语句里只写占位符，比如问号，真正的值由数据库驱动单独"
        "传递。这样无论用户输入什么内容，数据库都只会把它当作普通数据，而不会当作命令执行。"),
    _zh("zh_normalize_cosine",
        "余弦相似度衡量的是两个向量方向的接近程度，公式是点积除以两个向量长度的乘积。如果事先把每个向量都缩放成"
        "长度为一，分母就恒等于一，余弦相似度就等于点积本身。这样检索时只需要做乘法和加法，不必每次重新计算长度，"
        "而且所有分数都落在负一到一之间，方便比较。"),
    _zh("zh_http_422_400",
        "HTTP 400 表示请求本身有问题，服务器无法理解，比如 JSON 格式损坏。422 则表示格式可以解析，但内容不符合"
        "规则：字段缺失、类型错误或者数值超出允许范围。很多接口框架在参数校验失败时统一返回 422，并在响应里列出"
        "每个出错的字段，调用方据此就能知道要修改哪一项。"),
    _zh("zh_http_404_500",
        "状态码 404 表示服务器找不到请求的资源，例如文档编号不存在；500 表示服务器内部发生了意料之外的错误，"
        "与调用方的请求内容无关。"),
    _zh("zh_bot_token",
        "Telegram 机器人的令牌相当于它的密码，任何拿到令牌的人都能以这个机器人的身份收发消息。令牌应该放在 .env "
        "文件或环境变量里，并确保 .env 被 Git 忽略。如果令牌已经被推送到公开仓库，仅仅删除那次提交是不够的，因为"
        "历史记录和别人的副本里仍然存在；必须立刻通过 BotFather 撤销旧令牌并生成新令牌。"),
    _zh("zh_docker_volume",
        "容器的文件系统是临时的：容器一旦被删除，写在里面的数据库文件和上传文件都会跟着消失。需要长期保存的数据"
        "应该放在挂载卷或者绑定到宿主机的目录里，这样即使重新创建容器，数据仍然保留在宿主机上。"),
    _zh("zh_venv",
        "Python 虚拟环境为每个项目提供独立的软件包目录，不同项目即使依赖同一个库的不同版本，也不会互相干扰。"),
    _zh("zh_chunk_size_tradeoff",
        "块的大小没有放之四海而皆准的数值，它是在几个相互矛盾的目标之间做取舍。块太小时，一句话离开了上下文就"
        "可能失去意义，例如只剩下“它支持这种方式”，却看不出“它”指的是什么；检索到的片段虽然和问题相关，却不足以"
        "回答问题，后续生成答案时还需要拼接更多片段。块太大时，一个块里混杂了好几个主题，得到的向量是这些主题的"
        "平均，和任何一个具体问题的相似度都被稀释，最相关的那一段反而排不到前面。除此之外，嵌入模型能读取的长度"
        "有限，超过上限的部分会被直接忽略，块越大，被忽略的内容越多。对于中文文档，同样的字符数会产生更多的词元，"
        "这个问题更加明显。常见的做法是先选一个中等大小作为起点，然后用一组带有标准答案的问题分别测试几种大小，"
        "比较召回率和排名，而不是凭感觉决定。测试时要注意，块数变多以后，检索和存储的成本也会随之增加：每个块都"
        "需要单独计算和保存一个向量，精确检索的耗时与块数成正比。另外，重叠区域的长度通常随块的大小一起调整，"
        "例如块缩小一半时，重叠也相应缩短，否则重复存储的比例会越来越高。最后还要记住，评估集本身必须包含各种"
        "长度和语言的文档，否则得出的结论只适用于测试时恰好使用的那几篇文本，不能推广到真实用户上传的资料。\n\n"
        "还有一个容易被忽略的因素是文档本身的结构。说明书、合同和聊天记录的段落长度差别很大：说明书的一个段落往往"
        "只讲一个步骤，合同里的一个条款可能很长，而聊天记录每条消息都很短。按固定字符数切分时，前两者经常被从中间"
        "截断，后者又会把几十条无关的消息塞进同一个块。按段落边界切分可以缓解这个问题，但遇到特别长的段落时，仍然"
        "需要在句子或空格处再切一次。无论采用哪种方式，都应该保证同一份文档、同样的配置每次得到完全相同的块，这样"
        "块的编号才稳定，已经计算好的向量也不会因为重新切分而全部作废。"),
    _zh("zh_tokenization_cjk",
        "嵌入模型并不是按字或按词来读取文本，而是先用分词器把文本切成子词单元，也就是词元。多语言模型的词表需要"
        "同时覆盖上百种语言，常见的英文单词往往能整体对应一个词元，而中文里很多字、甚至常见的双字词都要单独占用"
        "一个词元。实际测量时，同样一千个字符，英文大约只有两三百个词元，中文却可能有六七百个。这意味着按字符数"
        "设定的块大小，对中文和英文的真实负担完全不同：一个一千二百字符的英文块离上限还很远，一个同样长度的中文块"
        "却已经超出。分词器还会在开头和结尾各加一个特殊标记，前缀“passage: ”本身也要占用几个词元，这些都计入上限。"
        "因此，讨论块的大小时，最好同时报告字符数和词元数，并且按语言分别统计，而不是用英文的经验比例去估算中文。"
        "分词器的结果是确定的：同样的文本和同样版本的词表，每次都会得到完全相同的词元序列，所以词元数可以放心地"
        "写进测试。反过来，如果升级了模型或者词表版本，词元数就可能变化，旧的统计结果需要重新测量，之前保存的"
        "向量也应当视为过期并重新生成，不能与新模型的向量混在一起比较。\n\n"
        "在实际项目里，可以用分词器对自己的语料做一次统计：分别取中文、英文和中英混合的文档，计算每一千个字符大约"
        "对应多少个词元，再据此判断当前的块大小会让多少比例的块超过上限。如果大部分中文块都被截断，就说明块的大小"
        "对中文来说偏大，需要在缩小块、按语言设置不同的块大小，或者换用上限更长的模型之间做选择。每一种选择都有代价："
        "缩小块会增加块的数量和检索成本，按语言区分会让切分逻辑更复杂，换模型则意味着所有向量都要重新生成，还要重新"
        "评估检索质量。\n\n"
        "还需要注意，词元的数量不仅取决于语言，也取决于文本的类型。代码片段、网址、长串数字和表格往往会被切成很多"
        "零碎的词元，同样的字符数可能比普通中文句子占用更多的名额；而常见的英文技术词汇通常只占一两个词元。因此，"
        "只用一种文本测出来的比例去推算整个语料库并不可靠，最好直接用分词器统计真实文档。"),
    _zh("zh_rag_hallucination",
        "语言模型生成答案时，有时会写出检索到的资料里根本没有的内容，这种现象通常称为幻觉。原因在于模型本质上是在"
        "续写最可能的文字，它会用训练时学到的常识去填补资料中的空白，而且语气同样肯定。降低风险的办法是在提示里"
        "明确要求只根据提供的片段作答，资料不足时直接说明不知道，并要求每一句结论都标注出处。标注出处之后，用户和"
        "测试程序都可以回到原文核对：如果某句话找不到对应的片段，或者对应的片段表达的是另一个意思，就说明这句话"
        "不可信。需要注意的是，出处标注本身也可能出错，模型可能引用了一个真实存在但并不支持该结论的片段，所以"
        "自动检查时不能只看有没有引用，还要看被引用的内容是否真的包含这个事实。\n\n"
        "幻觉在资料不完整时最容易出现。例如用户问一份合同的违约金比例，而检索到的片段只包含付款期限，模型仍然可能"
        "根据常见合同的写法给出一个看似合理的数字。又比如文档里写的是旧版本的配置，模型却按照它记忆中的新版本回答。"
        "这类错误特别危险，因为答案的语气和格式都很正常，用户很难察觉。所以当检索分数都偏低、或者片段之间互相矛盾时，"
        "更稳妥的做法是明确告诉用户资料不足，而不是勉强给出答案。\n\n"
        "在评估一个问答系统时，最好把检索和生成分开检查。先确认正确的片段有没有被检索出来，再判断模型有没有忠实地"
        "使用这些片段。如果正确片段根本不在检索结果里，再好的提示也无法让模型给出可靠的答案，这时应该改进切分、"
        "嵌入或排序，而不是修改提示词。反过来，如果片段已经在结果里，答案却仍然出错，才需要调整提示、减少一次放入"
        "的片段数量，或者要求模型先逐条列出依据再下结论。把每次测试用到的问题、检索结果和最终答案都保存下来，就能"
        "在修改之后逐条对比，看清楚改动到底改善了哪一类问题。"),
    _zh("zh_exact_search",
        "精确检索也叫暴力检索，做法非常直接：把问题的向量和数据库里每一个块的向量逐一计算相似度，然后按分数从高"
        "到低排序。它的最大优点是结果一定正确，不会漏掉真正最相似的块，而且实现简单，没有索引需要维护，新增或删除"
        "文档后立即生效。代价是计算量与块的数量成正比：块数乘以向量维度，就是每次检索需要做的乘法次数。\n\n"
        "具体实现时，先把所有候选向量叠成一个矩阵，每一行是一个块，每一列是一个维度，然后用这个矩阵乘以问题向量，"
        "一次运算就得到全部分数。矩阵运算由底层的数值库完成，比在 Python 里写循环逐个相乘快得多。对于三百八十四维"
        "的向量，几万个块在普通笔记本电脑上仍然只需要几毫秒到几十毫秒。内存方面，每个块的向量占用约一点五千字节，"
        "十万个块也只有一百多兆字节。\n\n"
        "如果检索时只想查某一篇文档，可以在读取候选向量时就按文档编号过滤，这样参与计算的行数更少。由旧模型生成的"
        "过期向量也应该在这一步被排除，而不是算完分数以后再删除，否则它们会占用计算时间，甚至混进结果里。\n\n"
        "为了让结果可以复现，排序时应该使用完整精度的分数，只在显示时四舍五入，分数完全相同的情况再按块的编号排序。"
        "在排序之前，还应该检查每个向量的长度是否为一、是否包含无效数值，因为一条损坏的数据就可能让整个排序失去意义。"
        "精确检索的结果还可以作为标准答案，用来衡量以后引入的近似方法漏掉了多少真正相关的块。\n\n"
        "在测试精确检索时，可以准备几个手工构造的小向量，事先算好它们与问题向量的点积，再检查程序返回的分数和顺序"
        "是否完全一致。还应该覆盖几种边界情况：数据库里一个块都没有时直接返回空结果，而不是加载模型；请求的数量大于"
        "块的总数时，只返回实际存在的块；指定的文档编号不存在时返回明确的错误。这些测试不依赖真实模型，运行得很快，"
        "可以在每次修改代码后反复执行。真实模型的测试则单独运行，用来确认前缀、维度和归一化这些约定在真实环境中依然"
        "成立。\n\n"
        "什么时候需要换成近似最近邻索引呢？经验上，当块的数量达到数十万甚至上百万，或者每秒要处理大量并发查询，逐一"
        "比较的延迟开始影响用户体验时，才值得引入专门的索引库或向量数据库。在那之前提前引入，只会增加部署和调试的"
        "复杂度。"),
    _zh("zh_deterministic_ties",
        "如果两个块的相似度分数完全相同，排序算法可能按照它们从数据库读出的先后顺序排列，而这个顺序并没有保证，"
        "于是同一个问题在不同时间会得到不同的结果顺序。解决办法是给排序增加一个固定的次要条件，例如分数相同时按"
        "块的编号从小到大排列，这样结果就完全可以复现，测试也不会时好时坏。"),
    _zh("zh_pdf_extraction",
        "从 PDF 中提取文字并不总是可靠。PDF 记录的是每个字符画在页面上的位置，而不是段落结构，所以分栏排版的文字"
        "可能被交错读出，表格的单元格也常常按坐标顺序连成一行，失去原来的行列关系。扫描件更特殊：页面上只有图片，"
        "没有文字层，普通的提取工具得到的是空白，必须先用光学字符识别把图片转换成文字。"),
    _zh("zh_batch_memory",
        "批量嵌入时，一次交给模型的块越多，需要同时保存的中间结果就越多。批次过大可能让内存占用突然升高，在只有"
        "十六吉字节内存的笔记本上甚至导致程序被系统终止；批次过小则会增加调用次数，总耗时略微变长。一般选择十几"
        "到几十之间的数值即可。"),
    _zh("zh_logging_privacy",
        "日志是排查问题的重要工具，但它也经常成为泄露信息的渠道。用户提出的问题可能包含姓名、电话、合同内容等"
        "隐私，机器人令牌和接口密钥更是绝对不能出现在日志里。比较稳妥的做法是只记录事件本身和统计信息，例如“检索"
        "完成，候选块一百二十个，耗时十五毫秒”，而不记录问题原文和返回的文本内容。\n\n"
        "确实需要记录请求内容时，应当先脱敏，比如只保留问题的哈希值，或者把手机号的中间几位替换成星号。异常堆栈"
        "也要小心，其中可能带有文件路径和变量取值，应该只写入服务器日志，不能直接返回给调用方。\n\n"
        "日志的级别也值得认真选择。调试级别的信息最详细，适合在本地开发时打开；线上环境一般只保留信息、警告和错误"
        "三个级别，否则日志量会迅速膨胀，真正重要的错误反而被淹没。为每个请求生成一个随机的请求编号，并把它写进"
        "这个请求产生的所有日志行，就能在不记录任何隐私内容的前提下，把同一次请求的各个步骤串联起来。采用结构化的"
        "格式，例如每行一条字段固定的记录，也方便以后用工具统计错误率和耗时。\n\n"
        "很多问题只有在用户真实使用时才会出现，所以线上日志不能完全关闭。比较好的折中是记录足够定位问题的信息："
        "哪个接口被调用、返回了什么状态码、处理了多少条数据、花了多长时间、是否发生了异常以及异常的类型。这些信息"
        "足以回答“最近是不是变慢了”“错误集中在哪个步骤”这类问题，却不会暴露用户具体问了什么。如果某个问题必须看到"
        "原始输入才能复现，应该在测试环境里用构造的数据重现，而不是从生产日志里翻找用户的真实内容。\n\n"
        "第三方服务也是容易被忽略的出口。把日志发送到外部的收集平台之前，要确认对方如何存储和保护这些数据，并在发送"
        "前过滤掉令牌、密码和个人信息。开发者在排查问题时随手把日志贴到聊天群或者公开的问题追踪页面，也可能造成泄露，"
        "因此贴出之前同样需要检查和删改。\n\n"
        "此外，日志本身需要规定保存期限：调试日志通常保留几天到几周即可，过期后自动删除；包含个人信息的记录保存得"
        "越久，被泄露的风险就越大。访问日志的权限也应当收紧，只让负责运维的人员查看，并记录谁在什么时候查看过。"),
    _zh("zh_stale_embeddings",
        "每个向量都只在生成它的那个模型下才有意义。更换嵌入模型、模型版本或输入前缀之后，旧向量和新问题的向量"
        "处在不同的空间里，直接比较得到的分数没有任何参考价值。因此存储向量时要同时记录模型名称和版本，检索时"
        "只使用与当前配置完全一致的向量，其余的视为过期，等待重新生成。"),
    # An FAQ packs unrelated topics into one long document, like many real
    # uploads. Its last answers lie beyond the 512-token window at 1200/200,
    # and (unlike the passages above) the visible part is about OTHER topics.
    _zh("zh_faq",
        "DocuBot 常见问题\n\n"
        "问：可以上传哪些类型的文件？\n答：目前支持纯文本 TXT、PDF 和 Word 的 DOCX 格式。旧版 Word 的 DOC 格式、"
        "图片和压缩包都会被拒绝，上传前请先转换成支持的格式。\n\n"
        "问：单个文件最大可以多大？\n答：单个文件不能超过十兆字节，超过限制的上传会直接失败并提示文件过大。如果文档"
        "确实很长，可以先按章节拆成几个文件再分别上传。\n\n"
        "问：文档处理失败了怎么办？\n答：先查看文档详情里的错误说明。常见原因是文件已损坏、PDF 设置了密码，或者文件"
        "里根本没有可以提取的文字。修正文件后重新上传即可，失败的记录不会影响其他文档。\n\n"
        "问：同一个文件上传两次会怎样？\n答：每次上传都会生成新的文档编号，系统不会自动合并重复的文件，所以同一份"
        "内容会被检索到两次。建议上传前确认是否已经存在。\n\n"
        "问：处理一篇文档需要多长时间？\n答：普通的文本文件通常几秒钟就能完成，页数很多的 PDF 需要更久。处理是同步"
        "进行的，请求返回时处理就已经结束。\n\n"
        "问：可以删除已经上传的文档吗？\n答：当前版本还没有提供删除接口。如果需要清空所有数据，可以在停止服务后删除"
        "存储目录里的内容。\n\n"
        "问：文档可以重新切分吗？\n答：可以。用新的块大小再次执行切分，旧的块会被整体替换，之前的向量也会失效，需要"
        "重新生成。\n\n"
        "问：一次最多返回多少条搜索结果？\n答：每次最多五十条，默认五条，结果按分数从高到低排列。\n\n"
        "问：支持中文文档吗？\n答：支持。嵌入模型是多语言模型，中文、英文以及中英混合的文本都可以处理，但很长的中文"
        "段落更容易超过模型的读取上限。\n\n"
        "问：搜索结果里的分数代表什么？\n答：分数是问题向量和文本块向量之间的余弦相似度，范围在负一到一之间，越大"
        "表示意思越接近。它不是概率，也不能在不同的模型之间直接比较。\n\n"
        "问：上传的文件保存在哪里？\n答：所有文件、处理结果和数据库都保存在项目目录下的 storage 文件夹里，不会发送"
        "到任何云端服务。备份时复制整个 storage 文件夹即可。\n\n"
        "问：可以在手机上通过聊天软件提问吗？\n答：通过 Telegram 提问的功能还在开发中。目前的机器人只会原样回复"
        "收到的消息，还不会根据文档内容回答问题。"),
    # --- Mixed Chinese-English -------------------------------------------------
    _mixed("mixed_fastapi_depends",
           "在 FastAPI 里，可以用 Depends 声明一个 dependency，比如 get_embedding_provider。每个 request "
           "到来时，FastAPI 会调用这个函数，并把返回值注入到 endpoint 的参数中。配合 functools.lru_cache，"
           "这个函数在整个 process 里只会真正创建一次 provider，所以 model 只 load 一次，之后所有 request "
           "都复用它。写 test 时，还可以通过 app.dependency_overrides 把真实 model 换成一个很小的 fake "
           "provider。"),
    _mixed("mixed_async_endpoints",
           "FastAPI 的 endpoint 可以写成 async def，也可以写成普通的 def。async def 适合等待 network 或 "
           "database 这类 I/O 的场景；但如果在 async def 里直接执行很耗 CPU 的计算，比如运行 embedding "
           "model，整个 event loop 都会被卡住，其他 request 只能排队。普通 def 的 endpoint 会被放进 thread "
           "pool 执行，不会阻塞 event loop，所以同步的 model 推理更适合写成普通 def。"),
    _mixed("mixed_truncation_512",
           "multilingual-e5-small 的 tokenizer 最多只处理 512 个 token，这个数字包含开头的 CLS、结尾的 "
           "SEP，以及 \"passage: \" 这个 prefix。超过 512 的部分会被直接丢弃，model 完全看不到，但它依然会"
           "返回一个看起来正常的 vector，不会报错。所以 pipeline 需要自己检查：tokenizer 返回的 overflowing "
           "不为空时，就说明发生了 truncation，DocuBot 会把这个 chunk 标记为 truncated=True 并在 API 里"
           "显示出来。对于 document 来说，截断是可以接受的折中，因为 chunk 的前半部分通常仍然包含主要内容，"
           "用户也能看到提示；但对于 query 来说，截断意味着用户以为整个问题都被搜索了，实际上 model 只看到了"
           "开头，所以更合理的做法是直接拒绝过长的 query，并告诉用户缩短问题。判断是否超过上限时，应该使用 "
           "model 自己的 tokenizer 精确计算，而不是用字符数乘以一个经验比例来估计，因为不同语言的比例差别很大。"
           "另外，这个检查应该复用已经 load 好的 tokenizer，不需要为了计数再 load 一次 model。\n\n"
           "在 chunking 这一侧，也可以利用同一个 tokenizer 做离线统计：对每种 chunk_size 分别计算 chunk 的 "
           "token 数分布，看有多少比例超过 512。英文 chunk 通常离上限很远，中文 chunk 更容易超出，mixed 的内容"
           "介于两者之间。统计结果应该和 retrieval 的评估一起看：truncated 的 chunk 如果检索效果明显变差，才有理由"
           "修改默认的 chunk_size；如果差别不大，保持现有配置反而更稳定，因为每次修改切分参数都会让已有的 "
           "embedding 全部失效，需要重新计算。"),
    _mixed("mixed_git_branch",
           "开发新功能时，可以先从 main 创建一个 feature branch，在上面 commit，完成后再通过 pull request "
           "合并回 main，这样 main 始终保持可以运行的状态。"),
    _mixed("mixed_webhook_https",
           "除了 long polling，Telegram bot 也可以使用 webhook：由 Telegram 主动把每条 update 通过 HTTPS "
           "POST 到你提供的 URL。这要求 server 有公网可访问的地址和有效的 TLS certificate，而且同一时间"
           "只能使用一种方式，设置 webhook 以后 getUpdates 就会失效。上线到云端时 webhook 更省资源。"),
    _mixed("mixed_embedding_cache",
           "第一次运行时，embedding model 的文件会从 Hugging Face 下载到 storage/model_cache 目录，之后"
           "直接从本地 cache 读取。只要 cache 完整，就可以设置 HF_HUB_OFFLINE=1 完全离线运行，不再访问 "
           "network。model 的 revision 是固定的 commit hash，所以每次拿到的都是完全相同的文件。"),
    _mixed("mixed_unicode_nfkc",
           "用户输入里常常混有全角字符，比如 ＡＰＩ 和 API 在屏幕上看起来差不多，但在 Unicode 里是完全不同"
           "的 code point，直接比较字符串会判断为不相等。使用 NFKC normalization 可以把全角字母和数字转换"
           "成对应的半角形式，这样后续的 matching 和统计才会一致。"),
)
