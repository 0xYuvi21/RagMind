"""
FastAPI backend for the RAG pipeline.

Previously this module imported a module-level `pipeline`/`_store` singleton
pair from Retrieve/llmquery.py and exposed one global, unauthenticated
`/api/chat` endpoint shared by every caller — no user, no per-chat isolation.

It now composes the auth and conversations routers (Backend/auth_router.py,
Backend/conversations_router.py), each built from Backend/deps.py's
Depends-based composition root, and initializes the SQLite schema on
startup. There is no direct import of a pipeline or vector store here at
all — every request resolves its own scoped dependencies.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.auth_router import router as auth_router
from Backend.conversations_router import router as conversations_router
from Backend.deps import init_app_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_app_db()
    yield


app = FastAPI(title="RagMind API", lifespan=lifespan)

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(conversations_router)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("Backend.backend:app", host="127.0.0.1", port=8000, reload=False)
