# CLAUDE.md

Changelog / onboarding doc for this rebuild. For design rationale (why each pattern was chosen), see
[SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) — this file only tracks *what changed*.

## What this project is

RagMind: a multi-tenant RAG chatbot. Students register, create conversations, upload notes
(PDF/DOCX/TXT/MD) mid-chat, and ask questions grounded in that conversation's own documents only.
JWT auth, per-conversation document isolation, and LangGraph-based short-term memory.

## How to run it

```bash
uv sync                                   # install/reconcile dependencies
cp .env.example .env                      # then fill in GROQ_API_KEY and JWT_SECRET_KEY at minimum

# Terminal 1 — Kafka broker (document ingestion queue; see "Async document
# ingestion via Kafka" below). Needs Docker installed.
docker compose up -d

# Terminal 2 — ingestion worker: consumes uploads, does the actual parsing/
# embedding. Without this running, uploads stay stuck at status "queued".
uv run python -m Backend.ingestion_worker

# Terminal 3 — the API. Use `python -m uvicorn`, NOT `uv run uvicorn` directly:
# `uv run uvicorn ...` hit "uv trampoline failed to canonicalize script path" in this environment;
# invoking uvicorn as a module sidesteps uv's console-script shim entirely.
uv run python -m uvicorn Backend.backend:app --reload --host 127.0.0.1 --port 8000

# Open Frontend/index.html directly in a browser (or serve it with any static
# file server) — it talks to http://127.0.0.1:8000 and handles CORS already.
```

Register a user, log in, create a conversation, upload a `.pdf`/`.docx`/`.txt`/`.md`, ask a question
about it. `data/app.db` (users/conversations/messages/ingestion jobs), `data/chroma/` (per-chat vector
stores), `data/checkpoints.sqlite` (LangGraph memory), and `data/uploads/` are all created on first run
and are gitignored.

## How to run the tests

```bash
uv run pytest tests/ -q
```

No live Groq key, no HuggingFace download, no network access required, **no Kafka broker required
either** — `tests/conftest.py` overrides every external dependency with a fake (see SYSTEM_DESIGN.md's
testing section), including `FakeIngestionProducer` (tests/fakes.py), which runs the real ingestion
pipeline synchronously in-process instead of round-tripping through a real queue/consumer. 57 tests, all
passing.

## Model/service configuration split: `config.yaml` vs `.env`

Per explicit user request, *which model or provider* to use for every swappable service (LLM,
embeddings, VLM, OCR, plus retrieval/chunking knobs) is now set in one file, **`config.yaml`** at the
project root — edit a value there and restart the server, no code change needed for any of it.
`.env`/`.env.example` is now reserved for secrets and environment-specific paths/URLs only (JWT key,
`DATABASE_URL`, `CHROMA_PERSIST_DIR`, `UPLOADS_DIR`, checkpoint settings, `GROQ_API_KEY`).

`Backend/config.py`'s `Settings` (still the single class every module receives config through) loads
`config.yaml` as an additional `pydantic-settings` source via `YamlConfigSettingsSource`, ranked below
env vars/`.env` — precedence highest-to-lowest: constructor kwargs (tests) > env var > `.env` >
`config.yaml` > field default. So an env var can still override a `config.yaml` value in a pinch, but
`config.yaml` is the intended place to change a model.

| `config.yaml` key | Purpose | Default |
|---|---|---|
| `groq_model` | Groq chat model name | `llama-3.3-70b-versatile` |
| `embedding_model_name` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `ollama_url` / `ollama_vlm_model` | Local Ollama endpoint for the VLM `read_image` tool | `http://localhost:11434/...` |
| `ocr_enabled` / `ocr_provider` | Whether OCR runs on embedded images, and with which backend | `true` / `rapidocr` |
| `retrieval_top_k`, `chunk_size`, `chunk_overlap` | Retrieval/chunking knobs | `5` / `512` / `100` |

## New environment variables (`.env` — secrets/paths only)

Only `GROQ_API_KEY` and `JWT_SECRET_KEY` need real values for a real deployment (everything else has a
sane default):

| Variable | Purpose | Default |
|---|---|---|
| `JWT_SECRET_KEY` | HMAC signing key for access tokens | dev placeholder — **must** be overridden |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` |
| `JWT_EXPIRE_MINUTES` | Access token lifetime (no refresh token — see SYSTEM_DESIGN.md) | 1440 (24h) |
| `DATABASE_URL` | SQLAlchemy URL for users/conversations/messages | `sqlite:///data/app.db` |
| `GROQ_API_KEY` | Groq API key for the chat LLM | none — chat endpoint fails gracefully without it |
| `CHROMA_PERSIST_DIR` | On-disk root for per-chat Chroma collections | `data/chroma` |
| `CHECKPOINTER_BACKEND` | `sqlite` (persistent) or `memory` (volatile, used by tests) | `sqlite` |
| `CHECKPOINT_DB_PATH` | SqliteSaver DB file | `data/checkpoints.sqlite` |
| `UPLOADS_DIR` | Where raw uploaded files (and images extracted from them) are stored per chat | `data/uploads` |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker address (see docker-compose.yml) | `localhost:9094` |
| `KAFKA_INGESTION_TOPIC` | Topic the upload endpoint publishes to / the worker consumes | `document-ingestion` |
| `KAFKA_CONSUMER_GROUP` | Consumer group id — run multiple `ingestion_worker.py` processes under the same group to spread load, none double-process a job | `ragmind-ingestion-workers` |
| `AUDIT_POSTGRES_HOST` / `_PORT` / `_DB` / `_USER` / `_PASSWORD` | Separate PostgreSQL database for audit logs (see "Audit logging" below) | all blank — audit logging is a no-op until these are filled in |

Removed (no longer applicable — see "Dropped `unstructured`" below): `UNSTRUCTURED_API_KEY`,
`UNSTRUCTURED_API_URL`. Moved from `.env` to `config.yaml` (see table above): `GROQ_MODEL`,
`EMBEDDING_MODEL_NAME`, `OLLAMA_URL`, `OLLAMA_VLM_MODEL`.

## Structural changes

### New files (didn't exist before)
- **Auth/DB/composition root**: `Backend/config.py`, `Backend/db.py`, `Backend/db_models.py`,
  `Backend/security.py`, `Backend/auth_dependencies.py`, `Backend/repositories.py`,
  `Backend/interfaces.py`, `Backend/schemas.py`, `Backend/deps.py`, `Backend/auth_service.py`,
  `Backend/conversation_service.py`, `Backend/auth_router.py`, `Backend/conversations_router.py`.
- **Per-format document parsers**: `RAG/parsers/` (`base.py`, `text_parser.py`, `markdown_parser.py`,
  `docx_parser.py`, `pdf_parser.py`) and `RAG/embeddings.py`.
- **OCR**: `RAG/ocr.py` — `OcrProvider` protocol + `RapidOcrProvider` (backed by
  `rapidocr-onnxruntime`), extracting the literal text of images embedded in PDF/DOCX uploads. Wired
  into `PdfParser`/`DocxParser` (both now take an optional `ocr_provider` constructor arg) and threaded
  through `DocumentIngestor` -> `VectorStore`/`VectorStoreFactory` -> `Backend/deps.py`. Produces a new
  `OCRText` element category (`content_type: "image_text"`), distinct from `Image`'s
  visual-description placeholder for the VLM tool. Off by default unless a provider is injected — bare
  `DocumentIngestor()`/`VectorStore()` calls (tests, `RAG/main.py`'s CLI demo) don't pay for loading the
  OCR engine; only the real `Backend/deps.py` composition root wires a real one in, gated by
  `config.yaml`'s `ocr_enabled`/`ocr_provider`.
- **Memory/checkpointing**: `Retrieve/checkpointer.py`.
- **Tests**: the entire `/tests` directory (`conftest.py`, `fakes.py`, `fixtures/mixed_content.py`,
  and one `test_*.py` per subsystem).
- **Config**: `config.yaml` — single source of truth for which model/provider each swappable service
  uses (see "Model/service configuration split" above).
- **Async ingestion via Kafka**: `docker-compose.yml` (single-node Kafka, KRaft mode),
  `Backend/kafka_producer.py` (`KafkaIngestionProducer`), `Backend/ingestion_worker.py` (standalone
  consumer process — run with `uv run python -m Backend.ingestion_worker`). See "Async document
  ingestion via Kafka" below for the full rationale and known limitations.
- **Audit logging**: `Backend/audit_db.py` (separate PostgreSQL engine/session, independent of
  `DATABASE_URL`), `Backend/audit_models.py` (one table per action category — `AuthAuditLog`,
  `ConversationAuditLog`, `MessageAuditLog`, `IngestionAuditLog`, `LLMEventAuditLog`,
  `VLMEventAuditLog` — all sharing an `AuditLogMixin`), `Backend/audit_logger.py`
  (`SqlAlchemyAuditLogger` + `NullAuditLogger`). See "Audit logging" below.
- **Docs**: this file, `SYSTEM_DESIGN.md`, `.env.example`, `.gitignore`.
- `Backend/__init__.py`, `RAG/__init__.py`, `Retrieve/__init__.py`, `VLM/__init__.py`,
  `RAG/parsers/__init__.py` — these four directories are now real Python packages (previously they
  relied on ad-hoc `sys.path.insert` hacks in each file); imports across the project are now proper
  `from RAG.ingest import ...` / `from Backend.deps import ...` absolute imports. Run everything from
  the project root (`uv run python -m RAG.main ...`, `uv run pytest`, `uv run python -m uvicorn
  Backend.backend:app`) so the root is on `sys.path`.

### Enhanced in place (same file, same class names, rewritten internals)
- **`Backend/backend.py`**: no longer imports a `pipeline`/`_store` singleton pair; now a thin FastAPI
  app factory that mounts `auth_router`/`conversations_router` and runs `init_app_db()` on startup via
  a `lifespan` hook. The old single global `POST /api/chat` endpoint is gone, replaced by
  conversation-scoped `/conversations/{id}/chat`.
- **`RAG/ingest.py`**: `DocumentIngestor` keeps the same public API (`ingest`, `ingest_file_stream`,
  `ingest_directory`, `supported_formats`) but now dispatches to `RAG/parsers/*` instead of
  `UnstructuredLoader` — see "Dropped `unstructured`" below. Now also takes an optional `ocr_provider`
  constructor arg (see the new `RAG/ocr.py` entry above) and builds its per-format parser instances
  itself (was a shared module-level `FORMAT_PARSERS` dict) so each `DocumentIngestor` can wire its own
  OCR provider into `PdfParser`/`DocxParser`. `ALLOWED_ELEMENT_CATEGORIES` gained `"OCRText"`.
- **`RAG/vector_store.py`**: `VectorStore` is otherwise unchanged (it already accepted an arbitrary
  `collection` name and `persist_dir`); added `VectorStoreFactory` (one collection per `chat_id`) and
  `VectorStore.drop()` (permanent delete, used when a conversation is deleted). `add_file`/
  `add_file_stream` gained an optional `image_output_dir` param, threaded through from
  `ConversationService` so images extracted from an upload land next to that upload. Both `VectorStore`
  and `VectorStoreFactory` now also take an optional `ocr_provider` param, passed straight through to
  their internal `DocumentIngestor`.
- **`RAG/chunking.py`**: unchanged behavior; `_md5` no longer takes an unused `idx` parameter it never
  actually hashed with (dead parameter removed).
- **`Retrieve/llmquery.py`**: `SimpleRAGPipeline` (a single retrieve-then-generate call, no memory) is
  gone. Replaced by `GroqLLMProvider` (an `LLMProvider` implementation), `build_retrieval_tool()` (a
  chat-scoped `search_knowledge_base` tool), and `build_chat_agent()` (a real LangGraph
  `create_react_agent`, checkpointed, with a `pre_model_hook` for history trimming). No module-level
  singletons; a CLI demo (`if __name__ == "__main__"`) builds everything locally for manual testing.
- **`VLM/imagellm.py`**: the module-level `read_image` `@tool` + `OLLAMA_URL`/`OLLAMA_MODEL` env-var
  constants are gone, replaced by `build_read_image_tool(ollama_url, ollama_model)` — a factory
  returning the tool bound to whichever endpoint/model is passed in, the same DI pattern
  `build_retrieval_tool()` already used. `Backend/conversation_service.py::ConversationService` now
  takes `ollama_url`/`ollama_vlm_model` constructor args (sourced from `config.yaml` via
  `Backend/conversations_router.py::get_conversation_service`) and calls the factory itself in
  `_optional_vlm_tool()`, instead of importing a ready-made module-level tool — so swapping the VLM
  model is a `config.yaml` edit, not a code change. Also gained optional `audit_logger`/
  `conversation_id`/`user_id` params on `build_read_image_tool()` — every `read_image` call is now
  recorded via `Backend/interfaces.py`'s `AuditLogger` (see "Audit logging" below); all three default
  to `None` so the CLI demo at the bottom of this file keeps working unaudited.
- **`Backend/config.py`**: `Settings` now also loads `config.yaml` (see "Model/service configuration
  split" above) via a custom `settings_customise_sources` adding `YamlConfigSettingsSource`; gained
  `ocr_enabled`/`ocr_provider` fields. Also gained `audit_postgres_host`/`_port`/`_db`/`_user`/
  `_password` (all blank by default) and a computed `audit_database_url` property that's `None` unless
  host/db/user are all set — see "Audit logging" below.
- **`Backend/deps.py`**: gained `_get_ocr_provider()` (builds a `RapidOcrProvider()` iff
  `settings.ocr_enabled`, else `None`), wired into `get_vector_store_factory()`. Also gained
  `get_ingestion_job_repository()` and `get_ingestion_producer()` (builds the real
  `KafkaIngestionProducer` from `settings.kafka_*`), and re-exports `get_session_factory` (was the
  private `_get_session_factory`) so `Backend/ingestion_worker.py` — a standalone process, not a route —
  can build DB sessions against the same engine/tables the web process uses. Also gained
  `get_audit_logger()` (returns `NullAuditLogger` unless `settings.audit_database_url` is set, else a
  `SqlAlchemyAuditLogger` against a lazily-built audit Postgres engine/session factory), and
  `init_app_db()` now also calls `init_audit_db()` when audit logging is configured.
- **`Backend/db_models.py`**: new `IngestionJob` model + `IngestionJobStatus` string constants
  (`queued`/`processing`/`done`/`failed`) — see "Async document ingestion via Kafka" below.
- **`Backend/repositories.py`** / **`Backend/interfaces.py`**: gained
  `SqlAlchemyIngestionJobRepository`/`IngestionJobRepository` (create/get/mark_processing/mark_done/
  mark_failed) and the `IngestionQueueProducer` protocol (`publish()`). `Backend/interfaces.py` also
  gained the `AuditLogger` protocol (`log_auth_event`/`log_conversation_event`/`log_message_event`/
  `log_ingestion_event`/`log_llm_event`/`log_vlm_event`) — see "Audit logging" below.
- **`Backend/schemas.py`**: `UploadResponse` (`file_name`, `chunks_added`) replaced by
  `IngestionJobResponse` (`id`, `file_name`, `status`, `chunks_added`, `error_message`, timestamps) —
  the upload endpoint's response shape changed (see below).
- **`Backend/conversation_service.py`**: `ingest_upload()` no longer parses/embeds inline — it
  persists the file, creates a `queued` `IngestionJob`, publishes a job reference to Kafka via the
  injected `ingestion_producer`, and returns immediately (extension validation still happens
  synchronously first, so an unsupported file type is still a fast 400, not a job that fails later).
  Gained `get_ingestion_job()` for status polling, and a new `IngestionJobNotFoundError` (same
  "don't confirm existence" 404 pattern as `ConversationNotFoundError`). Also takes an optional
  `audit_logger` constructor arg (defaults to `NullAuditLogger`) and now records a
  `ConversationAuditLog` row on create/rename/delete, an `IngestionAuditLog` row when a job is
  queued (and again if queueing itself fails), an `LLMEventAuditLog` row around every agent
  invocation in `ask()` (timed, with token usage pulled from the last message's `usage_metadata` when
  present), and a `MessageAuditLog` row per user/assistant turn — see "Audit logging" below.
- **`Backend/auth_service.py`**: `AuthService` now takes an optional `audit_logger` constructor arg
  and records an `AuthAuditLog` row for every register/login attempt, success or failure (a failed
  login records the attempted email with `user_id=None`, mirroring the "don't confirm existence"
  posture already used elsewhere in this codebase for conversation/job lookups).
- **`Backend/ingestion_worker.py`**: `_process_message()` now takes an `audit_logger` param (built once
  in `run()` via `deps.get_audit_logger()`) and records an `IngestionAuditLog` row on both the `done`
  and `failed` paths — the upload endpoint's own `IngestionAuditLog` row (queued) and the worker's
  (done/failed) together form the full lifecycle history for one job.
- **`Backend/conversations_router.py`**: `POST /conversations/{id}/upload` now returns
  `202 Accepted` + an `IngestionJobResponse` (was `200` + chunk count) — the document is **not**
  necessarily searchable in the response yet. New `GET /conversations/{id}/uploads/{job_id}` for
  polling job status, 404 if the job doesn't exist or belongs to another user's conversation.
- **`pyproject.toml`**: added `pyyaml` (config.yaml loading — already a transitive dep of
  `pydantic-settings`, now explicit), `rapidocr-onnxruntime` (OCR; pulls in `onnxruntime` — already
  present — plus `opencv-python`, `shapely`, `pyclipper`, `pillow`), `kafka-python` (Kafka
  producer/consumer client), and `psycopg2-binary` (PostgreSQL driver for the audit database).
- **`Frontend/index.html`**: same CSS/visual design; the JS data layer was replaced — real
  login/register (JWT in `localStorage`, sent as `Authorization: Bearer`), conversations fetched from
  `/conversations` instead of an in-memory `chats` object, per-chat file upload via
  `/conversations/{id}/upload`, chat via `/conversations/{id}/chat`, message history reloaded from
  `/conversations/{id}/messages` on switch (survives a refresh). Added a login/register overlay and a
  per-chat delete button, styled with the existing CSS custom properties. Upload flow now polls the new
  `GET .../uploads/{job_id}` endpoint (`pollUploadStatus()`, ~2s interval, gives up after ~30s) until a
  file is actually `done` before sending a chat question that might depend on it — otherwise a question
  asked immediately after upload could race the async ingestion and find nothing.

### Deleted
- `Backend/`, `RAG/`, `Retrieve/`, `VLM/` **directories were NOT deleted** — their files were enhanced
  in place per above. What *was* deleted:
  - `requirements.txt` — `pyproject.toml` + `uv.lock` is now the single dependency source of truth.
  - `RAG/chroma_db/` (the old global `rag_collection`, 475 chunks from the sample architecture PDF) —
    **deleted per explicit user decision** during this rebuild, not migrated (multi-tenancy makes a
    "global" collection meaningless; there was no principled owner to migrate it to).

### Left alone (out of scope)
`RAG/docs/` (ignored per user instruction — not used as a fixture or demo asset), root-level
`main.py`/`ocrtest.py`/`inspect.txt`/`project_summary.md`/`images.jpg`/`uploads/` — pre-existing scratch
files unrelated to the app, untouched.

## Dropped `unstructured` — per-format parsers instead

Originally this rebuild kept the existing `unstructured`-based ingestion pipeline (per the initial
task framing of "preserve working logic"). During development, real breakage in this environment led
to an explicit instruction to stop using `unstructured` (or any external parsing API) entirely and
write one parser per file format instead:

- `unstructured.partition.auto.partition()` **segfaulted** on a plain `.txt` file — traced to
  `python-magic` needing a native `libmagic` DLL that isn't present on Windows by default.
- Even after fixing that (`python-magic-bin`) and adding the full `unstructured[all-docs]` +
  `unstructured-inference` dependency chain (~80 extra packages: `torch`, `onnxruntime`, `opencv`,
  `pandas`, `pikepdf`, ...), `hi_res` PDF partitioning (needed for table-structure inference and image
  extraction) still requires **poppler** and **tesseract** system binaries not installed in this dev
  environment.

Replaced with `RAG/parsers/` — see SYSTEM_DESIGN.md's "Document parsing" section for the full
rationale and per-format details. Net effect: `uv sync` now resolves 164 packages instead of 243, no
native system binaries required, and no segfaults. Trade-off: a scanned/image-only PDF (the whole page
is one big image, no text layer at all) still produces no `NarrativeText` — this is a documented
limitation, not a silent failure. That trade-off narrowed since: embedded images *within* a page
(a diagram, a table screenshot, a photo of a whiteboard) now get OCR'd via `RAG/ocr.py`'s
`RapidOcrProvider` (`rapidocr-onnxruntime` — no `tesseract`/system binary needed either), producing an
`OCRText` chunk with the image's literal text. See the new-files entry above.

## VLM tool — wired in, not deferred

`VLM/imagellm.py`'s `read_image` tool (calls a local Ollama vision model, described visually — not
literal text; see the OCR entry above for that) is included in the agent's tool list
(`Backend/conversation_service.py::ConversationService._optional_vlm_tool()`) alongside
`search_knowledge_base`. It's loaded defensively (`try/except` around the import and around chat
invocation) since Ollama may not be running in every environment — if it isn't, the agent simply
doesn't have that tool available rather than the whole chat endpoint failing. When a PDF/DOCX image is
extracted during ingestion, its metadata includes an `image_path` the agent can pass to `read_image` if
a user asks about "the diagram"/"the image". Which Ollama endpoint/model it's bound to now comes from
`config.yaml` (see "Model/service configuration split" above), not a hardcoded/env-only constant.

## Async document ingestion via Kafka

**Motivation**: with multiple students uploading concurrently, synchronous ingestion (parse -> chunk ->
embed -> Chroma write, all inline in the upload request) has two real problems — (1) the embedding
model (`EMBEDDING_MODEL_NAME`, local HuggingFace, runs on CPU in-process) is a shared, CPU-bound
resource: concurrent uploads compete for it and all slow down together; (2) a from-scratch chat LLM call
would rate-limit under load too, though that's actually a *separate* concern (Groq, only on the `/chat`
path, not ingestion — see the conversation this was designed in). Decoupling ingestion from the request
via a queue means an upload returns instantly regardless of embedding load, and worker processes can be
scaled independently of the API.

**Architecture**: `POST /conversations/{id}/upload` persists the file to disk, creates an `IngestionJob`
row (status `queued`), and publishes `{job_id, conversation_id, user_id, file_name, stored_path}` to
Kafka (`Backend/kafka_producer.py`'s `KafkaIngestionProducer`) — the file's bytes never go through
Kafka itself, only a reference to where they already live on disk. `Backend/ingestion_worker.py`, run
as one or more separate processes sharing a consumer group, consumes those messages and runs the exact
same `VectorStoreFactory`/`DocumentIngestor` pipeline the endpoint used to run inline, then marks the
job `done` (with `chunks_added`) or `failed` (with an error message). The frontend polls
`GET /conversations/{id}/uploads/{job_id}` until it's done before asking a question that depends on it.

**Chosen over an in-process worker pool** specifically because the user wanted real Kafka experience
(interview/resume-relevant), not because it was the minimal fix for the stated problem — a bounded
semaphore/thread pool around the embedding call would have solved the CPU-contention problem with far
less operational overhead (no broker to run, no separate worker process, no async job-status API). If
that tradeoff ever stops being worth it, `IngestionQueueProducer` is a swappable protocol
(`Backend/interfaces.py`) — replacing Kafka with an in-process queue means writing one new class, not
touching `ConversationService`.

**Known limitations** (documented simplifications, not oversights):
- **No retry/backoff/dead-letter topic.** A failed job is marked `failed` with its error and left
  there; nothing automatically retries it. Acceptable for a learning/demo deployment, not
  production-grade delivery.
- **No cascade cleanup.** Deleting a conversation (`ConversationService.delete()`) doesn't delete its
  `IngestionJob` rows — they become harmless orphaned metadata (never queried again once the
  conversation's gone), not a correctness or security issue, just unswept rows.
- **~30s worst-case failure latency with no broker reachable.** `KafkaIngestionProducer` is tuned
  (`request_timeout_ms=5000, max_block_ms=8000`) to fail faster than kafka-python's ~30s default, but a
  live test against this environment (no Docker installed here — see below) still measured ~30s to
  return a `failed` job when no broker was listening at all, likely because the initial bootstrap
  connection attempt isn't fully bounded by those two settings in this kafka-python version. Once a
  broker is actually reachable, publish is fast (sub-second) — this only affects the "Kafka is down"
  degraded path, and even then, the request completes (202 + `failed` status) rather than hanging
  forever or 500ing.
- **Built without a local Docker install.** This dev environment has no `docker`/`docker compose`
  available (checked both PowerShell and the bash tool), so `docker-compose.yml`, the producer, and the
  consumer were written and unit/sanity-tested (imports cleanly, `publish()` fails gracefully and
  quickly against an unreachable address) but **not** run end-to-end against a real broker. The
  `docker-compose.yml` config is based on Apache Kafka's official KRaft single-node quickstart; if
  `docker compose up` fails, check `docker compose logs kafka` first — the most likely culprit is the
  `apache/kafka` image tag having moved on since this was written.
- **Client library**: `kafka-python` (pure Python, no C extension to compile — safer install on Windows
  than `confluent-kafka`, which needs `librdkafka`). Same producer/consumer/consumer-group concepts
  either way; swap later if a specific job posting wants `confluent-kafka` by name.

## Audit logging

**Motivation**: per explicit user request, every action in the system must be recorded to a SQL
database, with a separate table per action type, and the design must follow SOLID/OOP — no ad-hoc
`print`/log-file calls scattered through the codebase.

**Architecture**: a dedicated **PostgreSQL** database, entirely separate from the app's own
`DATABASE_URL` (which stays SQLite for users/conversations/messages) — `Backend/audit_db.py` builds its
own engine/session factory from `Settings.audit_database_url`, a computed property assembled from five
new blank-by-default fields (`audit_postgres_host`/`_port`/`_db`/`_user`/`_password`, see
`.env.example`). Six tables, one per action category, all in `Backend/audit_models.py`, sharing a common
`AuditLogMixin` (`id`, `user_id`, `occurred_at`) — this is the OOP "common base, specialized subclass"
shape: adding a new action category later is one small model class, not a schema change to a shared
polymorphic table:

| Table | Records |
|---|---|
| `audit_auth_events` | Every register/login attempt, success or failure (failed attempts keep the email, `user_id` null) |
| `audit_conversation_events` | Conversation create/rename/delete |
| `audit_message_events` | Every chat turn (user question + assistant answer), content length only — not the full text |
| `audit_ingestion_events` | Every ingestion job state change: queued (upload endpoint), done/failed (worker) |
| `audit_llm_events` | **Every Groq chat-model invocation** — model name, success, latency, and token usage when the model reports it (see below) |
| `audit_vlm_events` | Every `read_image` (Ollama) tool call — model, image path, success, latency |

`Backend/interfaces.py` gained an `AuditLogger` Protocol (Dependency Inversion — every call site depends
on this narrow interface, never on Postgres or SQLAlchemy directly) with one `log_*_event` method per
table above. `Backend/audit_logger.py` has the two implementations: `SqlAlchemyAuditLogger` (one INSERT
per event, its own short session per call, swallows and logs its own exceptions so a broken audit sink
never fails the user-facing action) and `NullAuditLogger` (no-op, satisfies the same Protocol — Liskov
substitution). `Backend/deps.py::get_audit_logger()` picks between them based solely on whether
`audit_database_url` resolves to a real value; **until real Postgres credentials are filled in, this is
a no-op and the app behaves exactly as before** — the credentials are deliberately left blank in
`.env.example` for you to fill in.

**Where events are recorded**:
- `Backend/auth_service.py::AuthService` — register/login, both outcomes.
- `Backend/conversation_service.py::ConversationService` — conversation CRUD, ingestion job queuing,
  chat messages, and the LLM call itself (wraps `agent.invoke()` in `ask()` with a timer; on success
  pulls `prompt_tokens`/`completion_tokens`/`total_tokens` from the last message's `usage_metadata` when
  the underlying model populates it — real `ChatGroq` does, `tests/fakes.py`'s fake model doesn't, so
  those fields are simply `None` in tests, which is correct, not a bug).
- `Backend/ingestion_worker.py::_process_message()` — the done/failed transition a real Kafka consumer
  produces (the upload endpoint already logs the initial `queued` transition).
- `VLM/imagellm.py::build_read_image_tool()` — gained optional `audit_logger`/`conversation_id`/
  `user_id` params, threaded in from `ConversationService._optional_vlm_tool()`; every `read_image` call
  is timed and logged regardless of success/failure.

**Why a separate Postgres database instead of new tables in the existing SQLite `app.db`**: matches
what was asked for (a SQL database, credentials supplied later) and keeps an ever-growing, write-heavy
audit trail off the same file the interactive app reads/writes on every request — a locked SQLite audit
table would otherwise contend with `app.db`'s own writes under load. `psycopg2-binary` was added to
`pyproject.toml` for this.

**Known limitations** (documented simplifications, not oversights):
- **No cross-database foreign keys.** `user_id`/`conversation_id`/`job_id` columns on audit tables are
  plain integers, not FKs — Postgres can't enforce a foreign key into a separate SQLite database. Same
  reasoning as `IngestionJob` rows not being cascade-deleted (see the Kafka section above): these are
  best-effort references for later querying, not referential-integrity-enforced.
- **Not run end-to-end against a real Postgres instance in this environment** (no Postgres server
  available here, and credentials are intentionally left blank for you to fill in) — `SqlAlchemyAuditLogger`
  was import/wiring-tested only, the same posture as the Kafka producer/consumer when this project was
  built without Docker (see above). Fill in `AUDIT_POSTGRES_*` in `.env`, run the app once, and
  `init_app_db()` will create all six tables via `Base.metadata.create_all` on startup.
- **No retention/rotation policy.** These tables grow forever; nothing here prunes old rows. Acceptable
  for a learning/demo deployment.

## Known environment quirks (this dev machine)

- `uv run uvicorn ...` fails with `uv trampoline failed to canonicalize script path`; use
  `uv run python -m uvicorn ...` instead (see "How to run it" above).
- No `poppler`/`tesseract` installed — irrelevant now that `unstructured` is gone, but relevant if
  OCR support is ever added back deliberately.
- No `docker`/`docker compose` installed — relevant to the Kafka ingestion queue above; the
  `docker-compose.yml`/producer/consumer were written and sanity-tested but not run end-to-end against
  a real broker in this environment.
