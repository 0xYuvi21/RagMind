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

# Run the API — use `python -m uvicorn`, NOT `uv run uvicorn` directly:
# `uv run uvicorn ...` hit "uv trampoline failed to canonicalize script path" in this environment;
# invoking uvicorn as a module sidesteps uv's console-script shim entirely.
uv run python -m uvicorn Backend.backend:app --reload --host 127.0.0.1 --port 8000

# Open Frontend/index.html directly in a browser (or serve it with any static
# file server) — it talks to http://127.0.0.1:8000 and handles CORS already.
```

Register a user, log in, create a conversation, upload a `.pdf`/`.docx`/`.txt`/`.md`, ask a question
about it. `data/app.db` (users/conversations/messages), `data/chroma/` (per-chat vector stores),
`data/checkpoints.sqlite` (LangGraph memory), and `data/uploads/` are all created on first run and are
gitignored.

## How to run the tests

```bash
uv run pytest tests/ -q
```

No live Groq key, no HuggingFace download, no network access required — `tests/conftest.py` overrides
every external dependency with a fake (see SYSTEM_DESIGN.md's testing section). 50 tests, all passing.

## New environment variables

All in `.env.example`; only `GROQ_API_KEY` and `JWT_SECRET_KEY` need real values for a real deployment
(everything else has a sane default):

| Variable | Purpose | Default |
|---|---|---|
| `JWT_SECRET_KEY` | HMAC signing key for access tokens | dev placeholder — **must** be overridden |
| `JWT_ALGORITHM` | JWT signing algorithm | `HS256` |
| `JWT_EXPIRE_MINUTES` | Access token lifetime (no refresh token — see SYSTEM_DESIGN.md) | 1440 (24h) |
| `DATABASE_URL` | SQLAlchemy URL for users/conversations/messages | `sqlite:///data/app.db` |
| `GROQ_API_KEY` | Groq API key for the chat LLM | none — chat endpoint fails gracefully without it |
| `GROQ_MODEL` | Groq model name | `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL_NAME` | HuggingFace embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | On-disk root for per-chat Chroma collections | `data/chroma` |
| `CHECKPOINTER_BACKEND` | `sqlite` (persistent) or `memory` (volatile, used by tests) | `sqlite` |
| `CHECKPOINT_DB_PATH` | SqliteSaver DB file | `data/checkpoints.sqlite` |
| `UPLOADS_DIR` | Where raw uploaded files (and images extracted from them) are stored per chat | `data/uploads` |
| `OLLAMA_URL` / `OLLAMA_VLM_MODEL` | Local Ollama endpoint for the VLM `read_image` tool | `http://localhost:11434/...` |

Removed (no longer applicable — see "Dropped `unstructured`" below): `UNSTRUCTURED_API_KEY`,
`UNSTRUCTURED_API_URL`.

## Structural changes

### New files (didn't exist before)
- **Auth/DB/composition root**: `Backend/config.py`, `Backend/db.py`, `Backend/db_models.py`,
  `Backend/security.py`, `Backend/auth_dependencies.py`, `Backend/repositories.py`,
  `Backend/interfaces.py`, `Backend/schemas.py`, `Backend/deps.py`, `Backend/auth_service.py`,
  `Backend/conversation_service.py`, `Backend/auth_router.py`, `Backend/conversations_router.py`.
- **Per-format document parsers**: `RAG/parsers/` (`base.py`, `text_parser.py`, `markdown_parser.py`,
  `docx_parser.py`, `pdf_parser.py`) and `RAG/embeddings.py`.
- **Memory/checkpointing**: `Retrieve/checkpointer.py`.
- **Tests**: the entire `/tests` directory (`conftest.py`, `fakes.py`, `fixtures/mixed_content.py`,
  and one `test_*.py` per subsystem).
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
  `UnstructuredLoader` — see "Dropped `unstructured`" below.
- **`RAG/vector_store.py`**: `VectorStore` is otherwise unchanged (it already accepted an arbitrary
  `collection` name and `persist_dir`); added `VectorStoreFactory` (one collection per `chat_id`) and
  `VectorStore.drop()` (permanent delete, used when a conversation is deleted). `add_file`/
  `add_file_stream` gained an optional `image_output_dir` param, threaded through from
  `ConversationService` so images extracted from an upload land next to that upload.
- **`RAG/chunking.py`**: unchanged behavior; `_md5` no longer takes an unused `idx` parameter it never
  actually hashed with (dead parameter removed).
- **`Retrieve/llmquery.py`**: `SimpleRAGPipeline` (a single retrieve-then-generate call, no memory) is
  gone. Replaced by `GroqLLMProvider` (an `LLMProvider` implementation), `build_retrieval_tool()` (a
  chat-scoped `search_knowledge_base` tool), and `build_chat_agent()` (a real LangGraph
  `create_react_agent`, checkpointed, with a `pre_model_hook` for history trimming). No module-level
  singletons; a CLI demo (`if __name__ == "__main__"`) builds everything locally for manual testing.
- **`VLM/imagellm.py`**: unchanged logic; `OLLAMA_URL`/`OLLAMA_MODEL` are now read from env vars
  (falling back to the original hardcoded defaults) instead of being hardcoded constants.
- **`Frontend/index.html`**: same CSS/visual design; the JS data layer was replaced — real
  login/register (JWT in `localStorage`, sent as `Authorization: Bearer`), conversations fetched from
  `/conversations` instead of an in-memory `chats` object, per-chat file upload via
  `/conversations/{id}/upload`, chat via `/conversations/{id}/chat`, message history reloaded from
  `/conversations/{id}/messages` on switch (survives a refresh). Added a login/register overlay and a
  per-chat delete button, styled with the existing CSS custom properties.

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
native system binaries required, and no segfaults. Trade-off: no OCR (no `tesseract`), so scanned/
image-only PDFs with no extractable text layer produce no `NarrativeText` — this is a documented
limitation, not a silent failure.

## VLM tool — wired in, not deferred

`VLM/imagellm.py`'s `read_image` tool (calls a local Ollama vision model) is included in the agent's
tool list (`Backend/conversation_service.py::ConversationService._optional_vlm_tool()`) alongside
`search_knowledge_base`. It's loaded defensively (`try/except` around the import and around chat
invocation) since Ollama may not be running in every environment — if it isn't, the agent simply
doesn't have that tool available rather than the whole chat endpoint failing. When a PDF/DOCX image is
extracted during ingestion, its metadata includes an `image_path` the agent can pass to `read_image` if
a user asks about "the diagram"/"the image".

## Known environment quirks (this dev machine)

- `uv run uvicorn ...` fails with `uv trampoline failed to canonicalize script path`; use
  `uv run python -m uvicorn ...` instead (see "How to run it" above).
- No `poppler`/`tesseract` installed — irrelevant now that `unstructured` is gone, but relevant if
  OCR support is ever added back deliberately.
