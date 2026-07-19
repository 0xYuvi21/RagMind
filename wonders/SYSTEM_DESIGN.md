# RagMind — System Design

This document explains the design of the production rebuild: what each concept is, why it was
chosen, and exactly where it's implemented. For a changelog of what moved/was added, see
[CLAUDE.md](CLAUDE.md).

## Layered architecture (API / service / repository / domain)

```
Backend/auth_router.py, Backend/conversations_router.py     ← API layer (FastAPI routes, HTTP concerns only)
Backend/auth_service.py, Backend/conversation_service.py     ← Service layer (business logic, orchestration)
Backend/repositories.py                                       ← Repository layer (SQLAlchemy data access)
Backend/db_models.py, Backend/interfaces.py                    ← Domain (ORM models, Protocols)
```

Routers never talk to SQLAlchemy or Chroma directly — they call a service (`AuthService`,
`ConversationService`), which calls repositories and the vector store/agent. This keeps HTTP status
codes and request/response shapes out of the business logic, and keeps ownership-checking logic
(a conversation must belong to the caller) in one place: `ConversationService.get_owned()`.

## Dependency Injection & Inversion of Control

**Where:** `Backend/deps.py` is the single composition root. Every route depends on a Protocol
(`Backend/interfaces.py`) via FastAPI's `Depends`, never on a concrete class constructed at import
time.

**Why:** The original `Backend/backend.py` did `from llmquery import pipeline, _store` — two
module-level singletons built once at import time, shared by every request regardless of user or
chat. That made per-chat isolation structurally impossible (one global store) and made testing
require a live Groq key and a populated global DB. `Backend/deps.py` instead exposes functions like
`get_llm_provider()`, `get_vector_store_factory()`, `get_checkpoint_backend()` that FastAPI resolves
per-request; `tests/conftest.py` overrides the exact same functions (`app.dependency_overrides[...]`)
to swap in a temp SQLite DB, a temp Chroma dir, and fakes for the LLM/embeddings — exercising the real
wiring path, not a parallel one.

## Repository pattern

**Where:** `Backend/repositories.py` — `SqlAlchemyUserRepository`, `SqlAlchemyConversationRepository`,
`SqlAlchemyMessageRepository`, each implementing the matching Protocol in `Backend/interfaces.py`.

**Why:** Isolates SQL/ORM details from the service layer. `ConversationService` calls
`self._conversations.get(id)` and doesn't know or care that it's backed by SQLAlchemy — swapping to a
different persistence engine only means writing new repository classes.

## Strategy pattern for swappable backends

**Where:** `Backend/interfaces.py` declares `EmbeddingProvider`, `LLMProvider`, `DocumentParser`
(implicitly, via `RAG/parsers/base.py`'s `FormatParser`), `VectorStoreBackend`/`VectorStoreFactoryProtocol`,
`CheckpointBackend`. Concrete implementations:

| Protocol | Production implementation | Test implementation |
|---|---|---|
| `EmbeddingProvider` | `RAG/embeddings.py::HuggingFaceEmbeddingProvider` | `tests/fakes.py::make_fake_embeddings` (hash-based `DeterministicFakeEmbedding`) |
| `LLMProvider` | `Retrieve/llmquery.py::GroqLLMProvider` | `tests/fakes.py::FakeLLMProvider` (scripted `ScriptedFakeChatModel`) |
| `FormatParser` | `RAG/parsers/{text,markdown,docx,pdf}_parser.py` | same classes — no fakes needed, they have no external dependencies |
| `CheckpointBackend` | `Retrieve/checkpointer.py::SqliteCheckpointBackend` | `Retrieve/checkpointer.py::InMemoryCheckpointBackend` |

**Why:** Every one of these is a place the project's constraints or environment might reasonably
change (a different LLM, a different embedding model, a new document format, a different persistence
backend for memory). None of them are ever imported directly by a router or service — always through
`Backend/deps.py`.

## Document parsing: no `unstructured`, no external API — one parser per format

**Where:** `RAG/parsers/base.py` (the `FormatParser` Protocol) and one class per extension:
`text_parser.py` (.txt), `markdown_parser.py` (.md), `docx_parser.py` (.docx, via `python-docx`),
`pdf_parser.py` (.pdf, via PyMuPDF/`fitz`). `RAG/ingest.py`'s `DocumentIngestor` dispatches by
extension (`FORMAT_PARSERS` dict) and enriches whatever a parser returns with shared metadata.

**Why this instead of `unstructured`:** The original prototype used
`langchain_unstructured.UnstructuredLoader` for every format. During this rebuild that dependency
proved both heavy and unreliable in a plain dev environment:
- A bare `unstructured.partition.auto.partition()` call **segfaulted** on a plain `.txt` file — traced
  to `python-magic` needing a native `libmagic` binary Windows doesn't ship (fixed empirically by
  installing `python-magic-bin`, but this class of native-dependency fragility is exactly what we
  wanted to avoid).
- `hi_res` PDF partitioning (needed for table-structure inference and image-block extraction) requires
  **poppler** and **tesseract** system binaries, neither present in this environment, plus ~80 extra
  Python packages (`torch`, `onnxruntime`, `opencv`, `pandas`, ...) just to import the module.
- Per explicit product decision, the project also avoids any *external/remote* partitioning API
  (`unstructured`'s hosted service) — everything must parse locally, deterministically, and
  auditable.

Each `FormatParser` instead uses one small, already-vetted library: `python-docx` for DOCX structure
(paragraphs, tables, and embedded image relationships are all native to the format's XML), PyMuPDF for
PDF (`page.get_text("blocks")` for text, `page.find_tables()` — PyMuPDF's own heuristic table
detector, no ML model — for tables, `page.get_images()` + `doc.extract_image()` for embedded images),
and plain file I/O + a small regex for TXT/MD. No system binaries, no multi-GB model downloads, no
segfaults. The trade-off: no OCR, so a scanned/image-only PDF with no text layer yields no
`NarrativeText` — documented, not silently swallowed (see CLAUDE.md).

Every parser tags each element's metadata with `category` ∈ {`NarrativeText`, `Table`, `Image`} — the
same three-way vocabulary the rest of the pipeline expects — so nothing downstream needed to change
when the parsing strategy changed. Table content is serialized as tab-separated rows (retrievable as
text); Image elements carry an `image_path` (or `image_ref` for Markdown) so the VLM tool
(`VLM/imagellm.py`'s `read_image`) can be pointed at the actual file. This is the fix for the original
bug where tables/images went missing — they were never being extracted as distinct elements in the
first place under the old strategy, not filtered out afterward.

## Multi-tenancy & data isolation strategy

**Where:** `RAG/vector_store.py`'s `VectorStoreFactory.get_or_create(chat_id)` — one Chroma
**collection per conversation** (`chat_<id>`), all under one shared `persist_dir`.

**Why this instead of a shared collection + metadata filter:** A shared collection with a mandatory
`chat_id` filter is one line of code away from a leak — any new route, tool, or bug that constructs a
query without that filter silently returns another chat's data. A separate collection per chat makes
isolation *structural*: a `VectorStoreBackend` instance physically has no reference to any other chat's
vectors, so there is no filter to forget. The trade-off is no cross-chat search (not a requirement
here) and many small collection directories on disk (fine at this project's scale — a classroom of
students, not a multi-tenant SaaS with millions of chats).

Proven by `tests/test_vector_store_isolation.py` (crafted-query leakage test at the store layer) and
`tests/test_chat_e2e.py`/`test_conversations_api.py` (the same guarantee through the full HTTP stack
and the agent's tool-calling path).

## JWT authentication & authorization flow

**Where:** `Backend/security.py` (bcrypt hashing + `PyJWT` encode/decode), `Backend/auth_dependencies.py`
(`get_current_user` FastAPI dependency), `Backend/auth_router.py` (`/auth/register`, `/auth/login`).

Flow: register hashes the password with `bcrypt` (called directly, not via `passlib` — `passlib`'s
last release predates bcrypt ≥4.x and emits a spurious version-detection warning against it; calling
`bcrypt` directly avoids the dependency and the warning) and stores the user. Login verifies the
password and issues one signed JWT (`sub` = user id, `HS256`, configurable expiry, default 24h) — no
separate refresh token. Every protected route depends on `get_current_user`, which decodes the bearer
token and loads the `User` row; conversation routes additionally call
`ConversationService.get_owned(conversation_id, current_user.id)`, raising `ConversationNotFoundError`
(→ HTTP 404) for both "doesn't exist" and "belongs to someone else" — a 404 rather than 403 so the
response doesn't confirm the id even exists.

**Why no refresh token:** a single 24h access token is enough for this project's scale (a
student-facing tool, not a long-lived enterprise session), and skipping refresh/rotation avoids
needing a token-revocation store. Trade-off: a user must log in again after 24h with no way to extend
a session silently. Documented here as a deliberate scope cut, not an oversight.

## Conversational memory / checkpointing strategy

**Where:** `Retrieve/checkpointer.py` (`SqliteCheckpointBackend`, `InMemoryCheckpointBackend`),
`Retrieve/llmquery.py` (`build_chat_agent`, `_trim_history`).

The agent is a LangGraph `create_react_agent` (`langgraph.prebuilt`) compiled with a
`BaseCheckpointSaver`. **`thread_id = str(conversation_id)`** at invoke time
(`Backend/conversation_service.py::ask()`) — this is what scopes memory to one conversation the same
way `VectorStoreFactory` scopes documents.

**Checkpointer backend:** `SqliteSaver` (via `langgraph-checkpoint-sqlite`) by default — conversation
state survives a server restart, which matters for a tool students return to across sessions.
`MemorySaver` is available via `CHECKPOINTER_BACKEND=memory`, used by the whole test suite so tests
never touch disk and run fast.

**Short-term memory window:** the checkpointer keeps the *full* thread (the source of truth, also
mirrored into the SQL `messages` table by `ConversationService.ask()` so the frontend's history
endpoint doesn't need to know anything about LangGraph internals) — but a `pre_model_hook`
(`_trim_history` in `Retrieve/llmquery.py`) trims what's actually sent to the LLM on each turn to the
last `MAX_HISTORY_MESSAGES` (12) via `langchain_core.messages.trim_messages`. This bounds latency/cost
per turn while keeping near-term pronoun/reference resolution ("it", "the previous file") working
within a conversation. Proven in `tests/test_agent_memory.py`: a follow-up question's model input is
asserted to contain the prior turn's content; a different conversation's model input is asserted to
never contain another conversation's content.

## Chunking strategy

**Where:** `RAG/chunking.py::DocumentChunker`, largely unchanged from the original prototype (it
already used sound defaults) — `RecursiveCharacterTextSplitter`, 512-character chunks with 100-character
overlap, splitting on paragraph/sentence/word boundaries in that preference order.

**Why these numbers:** 512 characters (~100-130 tokens) keeps chunks small enough for precise
retrieval (a chunk that mixes two unrelated ideas dilutes the embedding) while staying large enough to
retain useful context; 100-character overlap prevents a sentence that straddles a chunk boundary from
losing its context in both halves.

**Dedup:** every chunk's MD5(page_content) becomes its `chunk_id`; duplicate content within one
ingestion run is dropped, and — critically — `chunk_id` doubles as the Chroma document id (see
below), so this hash is what makes ingestion idempotent, not just non-duplicating within one call.

## Retrieval strategy

**Where:** `Retrieve/llmquery.py::build_retrieval_tool` wraps `VectorStore.similarity_search_with_score`
as a `search_knowledge_base` LangChain tool, `k=5` by default (`Backend/config.py::retrieval_top_k`),
bound to exactly one conversation's `VectorStore` instance (see multi-tenancy above). The agent decides
when to call it (a react-agent tool call), rather than every turn unconditionally retrieving —
follow-ups that don't need new context ("thanks!", "can you rephrase that") don't force a redundant
vector search.

## Idempotent ingestion

**Where:** `RAG/chunking.py`'s `chunk_id` (MD5 of chunk text) is used as the Chroma document id in
`RAG/vector_store.py::VectorStore._upsert()` (`self._db.add_documents(clean_chunks, ids=ids)`).

**Why it matters:** Chroma's `add_documents(ids=...)` upserts by id — re-ingesting the same file (a
student re-uploading a corrected version with mostly the same content, or a retry after a transient
failure) never creates duplicate vectors for unchanged chunks. At scale, without this, every retry or
re-upload would linearly bloat the collection and duplicate retrieved context in the LLM's prompt.

## Error handling & graceful degradation

- **LLM/agent failures:** `Backend/conversations_router.py::chat()` catches any exception from
  `ConversationService.ask()` and returns a friendly `{"text": "Sorry, I encountered an error...",
  "sources": []}` with HTTP 200 rather than a 500 — a transient Groq outage or a missing API key
  shouldn't break the chat UI. The real exception is still logged (`logger.exception(...)`) for
  debugging. Proven in `tests/test_chat_e2e.py::test_chat_with_missing_groq_key_degrades_gracefully`.
- **Missing Groq key doesn't block non-chat features:** `GroqLLMProvider.__init__` deliberately does
  *not* validate the API key (only `get_chat_model()` does, when actually invoked) — conversation
  create/list/rename/delete/upload must work even before `GROQ_API_KEY` is configured, since
  `get_llm_provider()` is a dependency of every conversation route, not just `/chat`.
  ingestion.
- **hi_res-style partitioning fallback pattern:** although the pipeline no longer uses `unstructured`,
  the general principle carries over — `RAG/parsers/pdf_parser.py`'s table/image extraction methods
  each catch and skip per-table/per-image failures (`except Exception: continue`) rather than failing
  the whole file over one malformed table or image.
- **Ownership vs. existence:** a conversation that doesn't exist and one that belongs to another user
  are indistinguishable from the API's perspective (both 404) — see JWT section above.

## Testing strategy

**Where:** `/tests`, run with `uv run pytest tests/`. `tests/conftest.py` overrides
`Backend/deps.py`'s dependency functions with a temp SQLite DB, a temp Chroma dir + hash-based fake
embeddings, an in-memory checkpointer, and a scripted fake chat model
(`tests/fakes.py::FakeLLMProvider`, built on LangChain's own `FakeMessagesListChatModel`) — no live
Groq key or HuggingFace download required to run the suite.

| File | Covers |
|---|---|
| `test_chunking.py` | size/overlap/dedup/chunk_id stability |
| `test_ingestion.py` | per-format parsing (.txt/.md/.docx/.pdf), the mixed-content (text+table+image) contract, metadata enrichment, unsupported/missing-file errors |
| `test_vector_store_isolation.py` | chat A vs. chat B — crafted-query leakage test, delete isolation |
| `test_jwt.py` | encode/decode roundtrip, expiry, tamper/wrong-secret/wrong-algorithm rejection |
| `test_auth_api.py` | register/login/duplicate-user/wrong-password/protected-route-401 |
| `test_conversations_api.py` | CRUD + cross-user ownership → 404 |
| `test_agent_memory.py` | same-thread follow-up sees prior turn; different thread doesn't; message persistence |
| `test_chat_e2e.py` | upload → ask → grounded answer; cross-chat leakage through the full stack; graceful degradation on LLM failure |

50 tests, all passing without external services. `test_ingestion.py` generates its fixtures at test
time (`tests/fixtures/mixed_content.py`, using `python-docx`/PyMuPDF directly — the same libraries the
parsers themselves use) rather than checking in binary files.
