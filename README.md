# docubot-ai
Production-oriented RAG Telegram knowledge assistant

> **Current status: Telegram Bot MVP + FastAPI file upload + document processing + chunking + local embeddings.**
> Uploaded documents can be turned into normalized text with source locations,
> split into deterministic chunks, and embedded into vectors with a local CPU
> model; everything is stored in SQLite.
> Vector search, retrieval, reranking, RAG answers, and other AI features are
> planned but **not implemented yet**.

## Current Features

**Telegram bot** ([app/bot.py](app/bot.py))

- `/start` command — replies `Welcome to DocuBot AI!`
- Normal-text Echo — replies `You said: <your message>`
- Environment-variable based token configuration (`TELEGRAM_BOT_TOKEN`)
- Local development using polling

**FastAPI backend** ([app/api.py](app/api.py))

- `GET /health` — health check
- `POST /upload` — upload one document
- `POST /documents/{document_id}/process` — extract and normalize its text
- `GET /documents/{document_id}` — processing status and metadata
- `POST /documents/{document_id}/chunk` — split the processed text into chunks
- `GET /documents/{document_id}/chunks` — list the stored chunks and their source locations
- `POST /documents/{document_id}/embed` — embed the chunks with a local model (CPU)
- `GET /documents/{document_id}/embeddings` — embedding status (valid / missing / stale)
- Supported file types: PDF, DOCX, TXT (max 10 MB)
- Local development storage in `storage/` (uploads, processed output, SQLite metadata)

**Local embeddings** ([app/embeddings/](app/embeddings/))

- Model `intfloat/multilingual-e5-small` (pinned revision) via FastEmbed, CPU only
- E5 input contract: chunks are embedded as `passage: <text>`, questions as `query: <text>`
- 384-dimensional, normalized float32 vectors stored in SQLite, up to 512 tokens per chunk
- Stale-vector detection (full embedding contract + text hash); re-running is idempotent

The Telegram bot and the web API are two **separate programs**. They are not
connected to each other yet.

## How It Works

Telegram bot:

```
Telegram User
→ Telegram
→ Python Bot (python-telegram-bot, polling)
→ Handler
→ Response
→ Telegram User
```

| User sends | Handler | Callback | Reply |
|---|---|---|---|
| `/start` | `CommandHandler("start")` | `start()` | `Welcome to DocuBot AI!` |
| Normal text (not a command) | `MessageHandler(filters.TEXT & ~filters.COMMAND)` | `echo()` | `You said: <text>` |

Web API upload:

```
Client (browser / script)
→ HTTP POST /upload (multipart/form-data, field name "file")
→ FastAPI validates filename, extension, size
→ File saved as storage/uploads/<random-id>.<ext>
→ Metadata row saved in SQLite (status "uploaded")
→ JSON metadata returned to the client
```

## API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /upload`

Send one file as `multipart/form-data` in a field named `file`.

Success — `201 Created`:

```json
{
  "file_id": "8adc0251f64d4cf983e29ce470705707",
  "filename": "handbook.pdf",
  "stored_filename": "8adc0251f64d4cf983e29ce470705707.pdf",
  "extension": ".pdf",
  "content_type": "application/pdf",
  "size_bytes": 12345,
  "status": "stored"
}
```

Errors return `{"detail": "..."}`:

| Status | When |
|---|---|
| `400 Bad Request` | Filename missing, too long, contains a path (`/` or `\`), or the file is empty |
| `413 Content Too Large` | File is larger than 10 MB |
| `415 Unsupported Media Type` | Extension is not `.pdf`, `.docx`, or `.txt` |
| `422 Unprocessable Content` | No `file` field was sent (FastAPI's built-in validation) |
| `500 Internal Server Error` | The file could not be saved, or an unexpected error happened (no internal details are returned) |

The `file_id` is the document's ID for the document endpoints below.

### `POST /documents/{document_id}/process`

Extracts and normalizes the text of an uploaded document and saves it as JSON
in `storage/processed/<document_id>.json`. Processing is **synchronous**: the
response arrives when processing is finished. Processing a document again is
allowed and produces the same output.

Success — `200 OK`:

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "original_filename": "handbook.pdf",
  "file_type": ".pdf",
  "content_type": "application/pdf",
  "size_bytes": 12345,
  "status": "completed",
  "created_at": "2026-09-24T01:30:00.123456+00:00",
  "updated_at": "2026-09-24T01:30:01.654321+00:00",
  "processed_path": "8adc0251f64d4cf983e29ce470705707.json",
  "error_message": null,
  "section_count": 12
}
```

`processed_path` is a filename inside `storage/processed/`. The extracted text
itself is never returned by the API.

Errors return `{"detail": "..."}`. When processing starts but fails, the
document's status becomes `failed` and the same safe message is stored in
`error_message`:

| Status | When |
|---|---|
| `400 Bad Request` | The document ID is not a valid ID (32 lower-case hex characters) |
| `404 Not Found` | No document with this ID, or its stored file is missing |
| `409 Conflict` | The document is already being processed |
| `415 Unsupported Media Type` | The file type has no parser |
| `422 Unprocessable Content` | Corrupted PDF, invalid DOCX, DOCX that unpacks to more than 50 MB, TXT not valid UTF-8, encrypted PDF, or no text found |
| `500 Internal Server Error` | Processed output or the database could not be written, or an unexpected error happened (no internal details are returned) |

### `GET /documents/{document_id}`

Returns the same metadata fields as above (without `section_count`):
`status` is one of `uploaded`, `processing`, `completed`, `failed`.
`processed_path` is set only when completed; `error_message` only when failed.
It returns `400` for an invalid ID and `404` for an unknown document.

### `POST /documents/{document_id}/chunk`

Splits a **processed** (`completed`) document into chunks and stores them in
SQLite (see [Chunking](#chunking)). Optional query parameters:

| Parameter | Default | Rule |
|---|---|---|
| `chunk_size` | `1200` | 1 to 10000 characters |
| `chunk_overlap` | `200` | 0 or more, and at most half of `chunk_size` |

```
POST /documents/8adc0251f64d4cf983e29ce470705707/chunk?chunk_size=1200&chunk_overlap=200
```

Success — `200 OK` (a summary only; the chunk text is not returned):

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "chunking_status": "chunked",
  "chunk_count": 5,
  "chunking_version": 1,
  "chunk_size": 1200,
  "chunk_overlap": 200,
  "chunked_at": "2026-09-24T02:10:00.123456+00:00"
}
```

Chunking the same document again **replaces** its previous chunks; it never
creates duplicates. Chunking does not change the document's processing `status`.

Errors return `{"detail": "..."}`:

| Status | When |
|---|---|
| `400 Bad Request` | The document ID is not valid |
| `404 Not Found` | No document with this ID, or its processed output file is missing |
| `409 Conflict` | The document is not processed yet (`uploaded`, `processing`, `failed`), or it was re-processed while being chunked |
| `422 Unprocessable Content` | Invalid `chunk_size` / `chunk_overlap`, or the processed document contains no text |
| `500 Internal Server Error` | The processed output is corrupted, the database could not be written, or an unexpected error happened (no internal details are returned) |

If chunking fails, the previously stored chunks (if any) are left unchanged.

### `GET /documents/{document_id}/chunks`

Lists a document's stored chunks in order. Add `?include_text=true` to also
return each chunk's text; by default `text` is `null` to keep the response small.

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "chunking_status": "chunked",
  "chunk_count": 5,
  "chunking_version": 1,
  "chunk_size": 1200,
  "chunk_overlap": 200,
  "chunked_at": "2026-09-24T02:10:00.123456+00:00",
  "source_filename": "handbook.pdf",
  "file_type": ".pdf",
  "chunks": [
    {
      "chunk_id": "8adc0251f64d4cf983e29ce470705707_v1_s1200_o200_00000",
      "chunk_index": 0,
      "char_count": 1143,
      "source_locations": [
        {"page": 1, "char_start": 0, "char_end": 820},
        {"page": 2, "char_start": 0, "char_end": 321}
      ],
      "text": null
    }
  ]
}
```

A document without chunks returns `"chunking_status": "not_chunked"`,
`"chunk_count": 0` and an empty `chunks` list. It returns `400` for an invalid
ID and `404` for an unknown document.

### `POST /documents/{document_id}/embed`

Embeds a chunked document's chunks with the local model (see
[Embeddings](#embeddings)). Only chunks without a valid embedding are
embedded, so calling it again is cheap and never creates duplicates. Optional
query parameter `batch_size` (1 to 64, default 16). The first call after the
server starts loads the model (a few seconds); the very first call on a new
computer also downloads it once (~490 MB).

Success — `200 OK` (vectors are never returned):

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "status": "complete",
  "model_name": "intfloat/multilingual-e5-small",
  "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
  "embedding_version": 2,
  "dimension": 384,
  "max_tokens": 512,
  "passage_prefix": "passage: ",
  "query_prefix": "query: ",
  "dtype": "float32",
  "normalized": true,
  "total_chunks": 5,
  "embedded_count": 5,
  "skipped_count": 0,
  "stale_reembedded_count": 0
}
```

`embedded_count` = chunks embedded in this run; `skipped_count` = chunks that
already had a valid embedding; `stale_reembedded_count` = the part of
`embedded_count` that replaced stale vectors.

Errors return `{"detail": "..."}`:

| Status | When |
|---|---|
| `400 Bad Request` | The document ID is not valid |
| `404 Not Found` | No document with this ID |
| `409 Conflict` | Not processed yet, not chunked yet, or re-chunked/re-processed while being embedded |
| `422 Unprocessable Content` | Invalid `batch_size` |
| `503 Service Unavailable` | The model could not be loaded (e.g. no internet on the very first download) |
| `500 Internal Server Error` | The model failed, the model files do not match the expected model configuration, stored data is corrupted, the database could not be written, or an unexpected error happened (no internal details are returned) |

### `GET /documents/{document_id}/embeddings`

Shows the embedding state for the **currently configured** model, without
loading the model and without returning vectors:

```json
{
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "status": "incomplete",
  "model_name": "intfloat/multilingual-e5-small",
  "model_revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
  "embedding_version": 2,
  "dimension": 384,
  "max_tokens": 512,
  "passage_prefix": "passage: ",
  "query_prefix": "query: ",
  "dtype": "float32",
  "normalized": true,
  "total_chunks": 5,
  "embedded_chunks": 3,
  "missing_embeddings": 1,
  "stale_embeddings": 1,
  "truncated_chunks": 0
}
```

`status` is `complete` (every chunk has a valid embedding), `incomplete`, or
`no_chunks` (the document is not chunked). `stale_embeddings` counts stored
vectors made under another contract (e.g. the old Batch 5 model) or from older
chunk text; they are never counted as valid. `truncated_chunks` counts valid
embeddings whose chunk was longer than the model can read (see
[Limitations](#embedding-limitations)). It returns `400` for an invalid ID and
`404` for an unknown document.

Interactive API docs are available at `http://127.0.0.1:8000/docs` while the server runs.

## File Upload Rules

- Supported: **PDF**, **DOCX**, **TXT** (checked by file extension, case-insensitive)
- Maximum size: **10 MB** (`MAX_UPLOAD_SIZE_BYTES` in `app/api.py`)
- The server generates the stored filename (`<random-id>.<ext>`). The user's
  original filename is only returned as metadata; it is never used as a file path.
- Files are stored locally in `storage/uploads/`. Uploaded files are ignored by Git.
- Uploading only stores the file. Text is extracted when the process endpoint
  is called, chunks when the chunk endpoint is called, and vectors when the
  embed endpoint is called. Documents are **not indexed or searchable** yet.

### Security limitations (development only)

This upload endpoint has basic validation, but it is **not production-secure** yet:

- The file type is checked only by **extension**. File contents are not
  inspected, so a renamed file would be accepted.
- There is **no authentication** — anyone who can reach the server can upload.
- There is **no virus/malware scanning**.
- The whole request is received before the 10 MB check runs. A production
  deployment should also limit request size at a reverse proxy.
- There is no rate limiting and no storage quota.
- Uploaded files are only stored, never executed.

## Document Processing

Code: [app/document_processing/](app/document_processing/)

```
Upload (POST /upload)
→ Process (POST /documents/{id}/process)
→ Extract text with the parser for the file type
→ Normalize whitespace
→ Store the processed representation (storage/processed/<id>.json)
→ Track status in SQLite (storage/metadata/documents.db)
```

| File | Responsibility |
|---|---|
| `models.py` | Data shapes: `Section`, `ProcessedDocument`, `Chunk`, `ChunkSet`, statuses, error types |
| `parsers.py` | Text extraction per file type + normalization (bytes in, sections out) |
| `chunking.py` | Chunking algorithm and its configuration (sections in, chunks out; no I/O) |
| `database.py` | SQLite tables `documents`, `chunks` and `embeddings` (schema), status changes, chunk storage |
| `processor.py` | The pipelines: processing (parse → save JSON) and chunking (load JSON → chunk → store) |

### Supported formats

| Format | Library | How text is split | Source location |
|---|---|---|---|
| PDF | `pypdf` | One section per page that has text | `{"page": 3}` (real page number) |
| DOCX | `python-docx` | One section per non-empty body paragraph, in document order | `{"paragraph": 12}` |
| TXT | Python standard library | One section per block of lines separated by blank lines | `{"block": 2, "line_start": 5, "line_end": 9}` |

TXT files must be UTF-8 (a UTF-8 BOM is allowed and removed).

### Processed representation

Every format produces the same JSON shape, so a later stage can read any
document the same way:

```json
{
  "schema_version": 1,
  "document_id": "8adc0251f64d4cf983e29ce470705707",
  "source_filename": "handbook.pdf",
  "file_type": ".pdf",
  "sections": [
    {"source_location": {"page": 1}, "text": "Welcome to the handbook."},
    {"source_location": {"page": 3}, "text": "Page three: rules."}
  ]
}
```

The output is deterministic: processing the same file again produces
byte-for-byte the same JSON (no timestamps inside, fixed key order).

### Normalization

Only whitespace is cleaned; the text is never rewritten or summarized:

- line endings become `\n`; null characters are removed
- runs of spaces/tabs become one space; each line is trimmed
- paragraph breaks are kept (more than one blank line becomes one blank line)
- letters, digits, and punctuation are never changed

### Processing status

```
uploaded → processing → completed
                      ↘ failed
```

Statuses are defined once in `DocumentStatus` and enforced by a SQLite
`CHECK` constraint. If processing fails, the status becomes `failed` with a
short, safe `error_message` (never a stack trace or file path). A `completed`
or `failed` document can be processed again. If the server is killed during
processing, the document can be processed again after 10 minutes instead of
staying stuck in `processing`. Processing a document again deletes its chunks
(they were made from the old processed output); chunk it again afterwards.

Files uploaded before metadata was recorded (older uploads) are registered
automatically the first time their ID is used; their original filename is
unknown, so the stored filename is shown instead.

### Limitations

- **No OCR**: scanned or image-only PDFs have no machine-readable text. They
  fail with a clear "No machine-readable text was found" message.
- **DOCX page numbers are not invented**: Word files have no fixed pages
  without rendering them, so DOCX locations are paragraph numbers.
- DOCX tables, headers, footers, and footnotes are not extracted yet.
- Encrypted PDFs are not supported.
- A DOCX file is a ZIP archive. DOCX files whose contents unpack to more than
  50 MB are refused, to protect the server from "zip bomb" files.
- Processing is **synchronous** in this MVP: the request waits until it finishes.
- SQLite is **local development** metadata storage (one file, one machine).
- All files (uploads, processed JSON, database) are stored **locally** and
  ignored by Git.
- Uploaded files are treated as untrusted data: they are only read and parsed,
  never executed, and extracted text is never treated as instructions.

## Chunking

Code: [app/document_processing/chunking.py](app/document_processing/chunking.py)

A **chunk** is a small piece of a processed document. Later stages
(Embedding → Vector Search → RAG) will work on chunks, not whole documents:
an embedding model can only take a limited amount of text, and a small,
focused piece of text is easier to find and to cite than a whole book.
Chunks can be embedded (see [Embeddings](#embeddings)); nothing searches
them yet.

```
Processed document (storage/processed/<id>.json, status "completed")
→ POST /documents/{id}/chunk
→ Load and validate the processed JSON
→ Build chunks (deterministic, no LLM, no network)
→ Replace the document's chunks in SQLite (one transaction)
```

### How documents are split

Sizes are measured in **characters**. The defaults are defined once in
`chunking.py` (`DEFAULT_CHUNK_SIZE = 1200`, `DEFAULT_CHUNK_OVERLAP = 200`) and
can be overridden per request.

1. The **blocks** are the Batch 3 sections: PDF pages, DOCX paragraphs, TXT
   blocks. Whitespace-only blocks are skipped.
2. **Small blocks are packed** into one chunk (joined by a blank line, `\n\n`)
   until the next block would make the chunk longer than `chunk_size`. Such
   blocks are never cut.
3. **Overlap between packed chunks:** a new chunk starts with the last whole
   block(s) of the previous chunk that fit into `chunk_overlap` characters.
   A block longer than the overlap is not repeated, so the real overlap can be
   smaller than `chunk_overlap`, or zero.
4. **A block longer than `chunk_size`** (e.g. a long PDF page) is split on its
   own into windows of at most `chunk_size` characters. A window ends at the
   last space or line break in its second half, so words stay whole (only a
   single "word" longer than that, like a very long URL, is cut in the middle).
   The next window starts up to `chunk_overlap` characters earlier, moved to
   the start of a word. Windows are never mixed with other blocks.
5. No chunk is ever empty, longer than `chunk_size`, or starts/ends with
   whitespace. Text is never rewritten or summarized: every chunk is made of
   exact slices of the processed text, and every non-whitespace character of
   the document is in at least one chunk.

The loops have explicit safety guards: every window must move forward, and a
chunk may never consist only of text the previous chunk already had.

**Why overlap is limited to half of `chunk_size`:** overlapping text is stored
twice. With an overlap close to `chunk_size`, every chunk would add only a few
new characters, and one document could turn into thousands of nearly identical
chunks (a denial-of-service risk found in security review). With the limit,
and because every window moves forward by at least a quarter of `chunk_size`,
the stored chunk text stays within a few times the document size, and
chunking time grows linearly with the document length.

**Limitation:** characters are **not** tokens. Embedding models measure input
in tokens. Measured with the current model's tokenizer: English ≈ 0.30 tokens
per character (1200 characters ≈ 360 tokens), mixed Chinese/English ≈ 0.41
(≈ 490 tokens), Chinese ≈ 0.64–0.69 (≈ 770–820 tokens). The model reads at most
**512 tokens**, so English and mixed chunks of the default size fit, but a full
1200-character **Chinese** chunk does not (see
[Embedding limitations](#embedding-limitations)).

### Source metadata

Every chunk lists **every** block it contains, in order, using the Batch 3
source location plus the exact character range used from that block's text
(`char_end` is exclusive, like Python slicing):

| Format | Example `source_locations` entry |
|---|---|
| PDF | `{"page": 3, "char_start": 0, "char_end": 812}` |
| DOCX | `{"paragraph": 12, "char_start": 0, "char_end": 240}` |
| TXT | `{"block": 2, "line_start": 5, "line_end": 9, "char_start": 0, "char_end": 318}` |

A chunk that spans pages lists each page. DOCX chunks never get page numbers
(Word files have no fixed pages). Together with `source_filename` and
`file_type`, this is what a later citation feature will use.

### Deterministic chunks and chunk IDs

The same processed document with the same `chunk_size` and `chunk_overlap`
always produces exactly the same chunks, in the same order, with the same IDs.
Nothing random, no timestamps and no LLM are involved. Chunk IDs are readable:

```
8adc0251f64d4cf983e29ce470705707_v1_s1200_o200_00003
```

= `<document_id>_v<chunking_version>_s<chunk_size>_o<chunk_overlap>_<chunk_index>`
(`chunk_index` starts at 0 and is padded to 5 digits).

- Different documents never share chunk IDs (the document ID comes first).
- Different settings give different IDs, so chunks made with different
  settings can never be mistaken for each other later.
- `CHUNKING_VERSION` is increased whenever the algorithm changes.
- `created_at` is stored as metadata only; it is **not** part of the ID.

### Storage and idempotency

Chunks are stored in the same SQLite database, in a `chunks` table:

| Column | Meaning |
|---|---|
| `chunk_id` | Deterministic ID (primary key) |
| `document_id` | Owning document (foreign key to `documents`) |
| `chunk_index` | Position in the document; unique per document |
| `text` | Exactly the text a later stage will embed |
| `char_count` | Length of `text` in characters |
| `source_locations` | JSON list, see above |
| `chunking_version`, `chunk_size`, `chunk_overlap` | Settings that produced the chunk |
| `created_at` | When the chunks were stored |

- The file name and type are not copied into every chunk; they are read from
  the `documents` table.
- Chunking again **replaces** the document's chunks in one transaction, so
  repeated calls never create duplicates (this is called being idempotent).
  `UNIQUE(document_id, chunk_index)` also makes duplicates impossible in the
  database itself.
- Chunking status is **separate** from processing status and is not stored as
  its own column: a document is `chunked` exactly when it has rows in `chunks`.
- If the document is re-processed while it is being chunked, the chunks are
  not saved (`409`), so stored chunks always match the current processed output.

### Current limitations

- Chunk size is counted in characters, not model tokens (see above).
- Chunking is synchronous and runs only when the endpoint is called. Normal
  documents take well under a second, but an extreme file (e.g. millions of
  one-character TXT blocks) can take tens of seconds.
- The whole chunk list is returned at once (no pagination yet).
- Re-chunking a document deletes its embeddings (they must be embedded again).

## Embeddings

Code: [app/embeddings/](app/embeddings/)

- **Embedding** — 把文本转换成向量，使系统可以比较语义相似度。
  Turning text into a list of numbers so that texts with similar meaning get
  similar numbers.
- **Vector** — 一串浮点数，表示文本在模型语义空间中的位置。
- **Dimension** — 向量有多少个数字。A **384-dimensional embedding** is a vector
  of 384 numbers, e.g. `[0.021, -0.113, 0.087, …]` (384 values).

Chunks need embeddings because a later search step (not built yet) will
compare a question's vector with every chunk's vector to find the chunks with
the closest meaning. This batch only **creates and stores** the vectors.

```
Chunks (SQLite, chunk_index order)
→ POST /documents/{id}/embed
→ Compare each chunk's text hash and the current contract with stored embeddings
→ Only missing or stale chunks → "passage: " + chunk text → local model (CPU), 16 per batch
→ Normalize to length 1, round to float32
→ Save each batch in one short SQLite transaction (chunk text itself is never changed)
```

| File | Responsibility |
|---|---|
| `config.py` | The one place that names the model, its pinned revision, prefixes and settings |
| `models.py` | Data shapes: `EmbeddingContract`, records, status, error types |
| `vectors.py` | Text hash, normalization, float32 bytes (standard library only) |
| `provider.py` | The local FastEmbed model and the pinned download; the only module that imports `fastembed` / `huggingface_hub` |
| `storage.py` | SQL for the `embeddings` table |
| `service.py` | The pipeline: chunks → model → SQLite, and the status report |

### The model

| | |
|---|---|
| Library | [FastEmbed](https://github.com/qdrant/fastembed) 0.8.1 (ONNX Runtime, CPU only, no GPU/CUDA, no PyTorch) |
| Model | [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small), file `onnx/model.onnx` (float32) |
| Revision | `614241f622f53c4eeff9890bdc4f31cfecc418b3` (exact Hugging Face commit, pinned) |
| License | MIT |
| Languages | 94 according to the model card, including Chinese and English |
| Dimension | 384 |
| Max input | 512 tokens, `passage: ` prefix included (longer text is cut; see limitations) |
| Pooling | Mean pooling over the tokens, then L2 normalization (as in the model card) |
| Size | ~470 MB on disk; ~0.8 GB RAM after loading, up to ~2 GB while embedding a batch of 16 long chunks |

All values were checked against the official repository files at that
revision (`config.json` → `hidden_size` 384, `tokenizer_config.json` →
`model_max_length` 512, `1_Pooling/config.json` → mean pooling). At load
time DocuBot checks the downloaded `config.json` and the tokenizer's real
token limit again and refuses files that do not match (`500`, safe message).

This model is the project's **current development baseline**, chosen because
it is multilingual (Chinese + English), made for retrieval, runs on a CPU and
reads 512 tokens (Batch 5's first model read only 128). It is **not** a
proven "best" model: nothing here measures retrieval quality yet; that needs
an evaluation set in a later batch.

**How it is loaded.** FastEmbed 0.8.1 does not list this model (only
`multilingual-e5-large`), so it is registered with FastEmbed's documented
`TextEmbedding.add_custom_model()` (mean pooling, 384 dimensions,
`onnx/model.onnx`). FastEmbed's own normalization is switched off; DocuBot
normalizes every vector itself, so `normalized` in the contract is always the
truth. The model name, revision, prefixes and limits are defined once, in
`app/embeddings/config.py` (`MULTILINGUAL_E5_SMALL`). Only models listed
there (`SUPPORTED_MODELS`) are accepted, so a configured name can never point
to arbitrary files.

### E5 input contract: `passage:` and `query:`

E5 models are trained for **asymmetric retrieval**: a short question and a long
document are different kinds of text, and the model must be told which one it
is reading. The model card requires:

| Text | Exact model input | Method |
|---|---|---|
| Document chunk | `passage: FastAPI is a Python web framework.` | `embed_documents()` |
| Question (future search) | `query: What is FastAPI?` | `embed_query()` |

- The prefix is added **only at embedding time**, in `provider.py`. The chunk
  text stored in the `chunks` table is never changed and never starts with
  `passage:`.
- FastEmbed adds no prefix of its own for this model, so there is exactly one
  place that adds it.
- Using the wrong prefix does not crash anything; it silently makes search
  worse. Tests therefore check that documents always use `passage: `, queries
  always use `query: `, and that the service refuses a vector whose input hash
  does not match `passage: ` + chunk text.
- Both kinds of vectors come from the same model and live in the same
  embedding space, so a query vector can later be compared with chunk vectors.

### Model cache and network

- The model files of the pinned revision are downloaded **once** from Hugging
  Face into `storage/model_cache/` the first time they are needed (with
  `huggingface_hub.snapshot_download(revision=...)`, then passed to FastEmbed
  as `specific_model_path`). This folder is ignored by Git; model files are
  never committed.
- Only five files are downloaded: `config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json` and `onnx/model.onnx`.
  No Python code, pickle or PyTorch files are fetched or executed.
- After that, loading uses only the local files; embedding never calls an
  external API (no OpenAI, Anthropic, Cohere, etc.). The smoke test verifies
  this with network access for Hugging Face switched off (`HF_HUB_OFFLINE=1`).
- The model is loaded once per server process and reused for every request,
  never once per chunk.
- Files of the old Batch 5 model (`models--qdrant--paraphrase-…`) may still be
  in the cache folder; they are no longer used and can be deleted by hand.
- **Supply-chain note:** model files are third-party inputs. The revision is
  pinned to an exact commit, so the same files are downloaded every time, but
  DocuBot does not compare its own checksums of the files. The integrity of
  the download is left to `huggingface_hub`, and a cached copy counts as
  complete when all five files exist (their sizes are not re-checked). If a
  cached file is ever damaged, delete `storage/model_cache/models--intfloat--multilingual-e5-small/`
  and it is downloaded again.

### Embedding contract and stale embeddings

A vector is only meaningful together with the exact process that produced it,
so every stored vector records its **contract**: `model_name`,
`model_revision`, `embedding_version`, `dimension`, `max_tokens`,
`passage_prefix`, `dtype` (`float32`) and `normalized`.

Two different texts are involved and are kept apart:

| Field | SHA-256 of | Used for |
|---|---|---|
| `text_sha256` | the **original** chunk text (as stored in `chunks`) | noticing that a chunk's text changed |
| `input_sha256` | the **exact model input**: `passage: ` + chunk text | proving which input produced the vector |

A stored embedding is **valid** only if every contract field matches the
current configuration and both hashes match the chunk's current text.
Otherwise it is **stale**: it is never treated as valid, the status endpoint
counts it, and the next embed run replaces it. So:

- chunk text changes → hashes change → re-embedded
- model, revision, `EMBEDDING_VERSION`, token limit, passage prefix or
  normalization changes → every vector is stale → re-embedded
- vectors from different contracts are never mixed or counted together
- changing only the **query** prefix does not make stored vectors stale
  (stored vectors are always passages)

`EMBEDDING_VERSION` is **2** since this batch (Batch 5A). Version 1 was
`paraphrase-multilingual-MiniLM-L12-v2` (128 tokens, no prefix).

**Normalization:** DocuBot scales every vector to length 1 itself
(`normalized = true`). Later, cosine similarity is then just a dot product.
Normalization only changes the maths; it says nothing about retrieval quality.

### Upgrading a Batch 5 database

A database created by Batch 5 is upgraded automatically the next time it is
opened: the four new columns (`model_revision`, `max_tokens`,
`passage_prefix`, `input_sha256`) are added with `ALTER TABLE`. Nothing is
deleted: chunk text and chunk IDs stay exactly the same, and the old
MiniLM vectors stay in the table, but their contract (old model, version 1,
no revision, no prefix) never matches, so they are reported as
`stale_embeddings` and are never used. Calling
`POST /documents/{id}/embed` replaces them with E5 vectors (one row per
chunk, no duplicates).

### Storage and idempotency

Vectors are stored in the existing SQLite database, table `embeddings`:

| Column | Meaning |
|---|---|
| `chunk_id` | The chunk (primary key and foreign key to `chunks`) |
| `model_name`, `model_revision`, `embedding_version`, `dimension`, `max_tokens`, `passage_prefix`, `dtype`, `normalized` | The contract |
| `text_sha256` | Hash of the original chunk text |
| `input_sha256` | Hash of the exact model input (`passage: ` + chunk text) |
| `truncated` | 1 if the model only read the first part of the input |
| `vector` | `dimension × 4` bytes: little-endian float32 BLOB (384 → 1536 bytes) |
| `created_at` | When the vector was stored (metadata only) |

- **One active embedding per chunk.** A replacement overwrites the old row
  (UPSERT); there are never hidden duplicate vectors.
- **Idempotent:** a second run with the same chunks and contract embeds
  nothing (all chunks are skipped).
- The model runs **outside** any database transaction. Each batch is then saved
  in one short transaction that re-checks that the chunk still exists and still
  has exactly the embedded text; otherwise the batch is not saved (`409`).
  If a run stops halfway, finished batches stay saved and the next run continues.
- Deleting a chunk (re-chunking or re-processing) also deletes its embedding
  (`ON DELETE CASCADE`), so a vector can never outlive its text.
- Corrupted rows (wrong BLOB size, NaN, wrong types) are rejected with a safe error.

### Determinism

The same model files, the same text and the same configuration produce the
same vector on the same computer and software versions (tested: repeated runs
and different batch sizes gave identical numbers). Floating-point results can
differ in the last digits across different CPUs or ONNX Runtime versions, so
bit-identical vectors on every machine are **not** guaranteed. Nothing random
and no timestamps influence a vector.

### Embedding limitations

- **Long Chinese chunks are still truncated.** The model reads at most 512
  tokens. Measured on this project's real chunks (default 1200 characters):
  English ≈ 360 tokens and mixed Chinese/English ≈ 410–490 tokens fit
  completely, but a full 1200-character Chinese chunk needs ≈ 770–820 tokens,
  so only roughly its first 750–800 characters influence the vector. Such
  chunks are flagged (`truncated_chunks`). In a benchmark of 47 real chunks,
  1 was truncated (the old 128-token model: 45). The Batch 4 chunk size was
  **not** changed in this batch (changing it would change every chunk ID);
  token-aware or language-aware chunking is an open decision.
- CPU speed on the development laptop (Intel i5-1334U, 16 GB RAM): loading the
  cached model takes about 2–4 s; embedding takes roughly 160 ms per
  default-size chunk (200 chunks ≈ 32 s), about 3× slower than the old
  128-token model, because each chunk is read completely. Peak memory while
  embedding batches of 16 long chunks was about 2 GB. The first download
  (~490 MB) took about 80 s on the development network.
- Embedding is synchronous; a large document keeps the request waiting.
- Re-chunking deletes embeddings, even if the chunk text did not change.
- Retrieval quality has **not** been evaluated; the model is a development
  baseline, not a benchmark winner.
- There is **no vector search, no vector database, no retrieval, no reranking,
  and no RAG answer generation** yet. Vectors are stored only.

## Local Setup

1. Python environment uses a local virtual environment in `.venv`.
2. Dependencies are listed in `requirements.txt`:

   ```powershell
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. The Telegram token is stored in a `.env` file in the project root.
   Copy `.env.example` to `.env` and put your own token from BotFather in it:

   ```
   TELEGRAM_BOT_TOKEN=your-token-here
   ```

4. `.env` must **never** be committed. It is already listed in `.gitignore`.

The web API does not need the Telegram token. The first embed request
downloads the embedding model (~490 MB, pinned revision) into
`storage/model_cache/`; this needs internet access once.

## Local Run

Run these from the project root.

Telegram bot:

```powershell
.venv\Scripts\python.exe -m app.bot
```

Web API (development server with auto-reload):

```powershell
.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

The API listens on `http://127.0.0.1:8000`. Press `Ctrl+C` to stop either program.

## Testing

Tests use Python's built-in `unittest`. They do not contact Telegram, do not
use the real `.env`, and write uploads, processed output, and SQLite databases
only into temporary folders. Test PDFs and DOCX files are generated in code
([tests/sample_documents.py](tests/sample_documents.py)):

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The normal test run never downloads or loads the real embedding model: a small
fake model ([tests/fake_embeddings.py](tests/fake_embeddings.py)) is used. A
separate smoke test uses the real model (it downloads it once if needed, then
runs offline):

```powershell
$env:DOCUBOT_RUN_MODEL_SMOKE_TEST = "1"
.venv\Scripts\python.exe -m unittest tests.test_embedding_smoke -v
```
