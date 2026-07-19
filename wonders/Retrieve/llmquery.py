"""
llmquery.py
===========

Builds the conversational RAG agent.

Previously this module built a `SimpleRAGPipeline` — a single retrieve-then-
generate call with no memory — and exposed it plus a global `VectorStore` as
module-level singletons (`pipeline`, `_store`) that Backend/backend.py
imported directly. That made every chat share one knowledge base and made the
"agent" untestable without a live Groq key and a populated global DB.

This version builds a real LangGraph react-agent (retrieval tool + VLM image
tool) compiled with a CheckpointBackend for short-term memory. Nothing here is
a module-level singleton: `build_chat_agent()` takes its LLM provider, tools,
and checkpointer as arguments, and the retrieval tool is bound to whichever
conversation's isolated VectorStore the caller passes in (see
RAG/vector_store.py's VectorStoreFactory and Backend/conversation_service.py,
which is the only place that ties a chat_id to a specific store + thread_id).

    User Query
        │
        ▼
    LangGraph react-agent  ── search_knowledge_base tool ──▶ this chat's VectorStore
        │                  └─ read_image tool ──▶ VLM/imagellm.py (Ollama), on demand
        ▼
    Final answer (checkpointed under thread_id = chat_id)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
RAG_DIR = BASE_DIR / "RAG"
sys.path.insert(0, str(RAG_DIR))

from dotenv import load_dotenv
from langchain_core.messages import trim_messages
from langchain_core.tools import StructuredTool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

from RAG.vector_store import VectorStore

load_dotenv(BASE_DIR / ".env")

DEFAULT_TOP_K = 5

# ─────────────────────────────────────────────────────────────────────────────
# Short-term memory policy
# ─────────────────────────────────────────────────────────────────────────────
# The checkpointer (Retrieve/checkpointer.py) keeps the FULL message thread —
# that's the source of truth, also mirrored into the SQL `messages` table
# (Backend/db_models.py) so the frontend's history endpoint doesn't depend on
# LangGraph internals. But sending an ever-growing thread to the LLM on every
# turn is unbounded latency/cost, so a pre_model_hook trims what's actually
# sent to the model to the last MAX_HISTORY_MESSAGES messages. This is enough
# to resolve near-term pronouns/references ("it", "the previous file") within
# a conversation without re-sending the entire history every turn.
MAX_HISTORY_MESSAGES = 12

SYSTEM_PROMPT = """
You are a helpful document assistant for a student's notes.

Use the `search_knowledge_base` tool to find relevant context from the documents
uploaded to THIS conversation before answering a question about the user's notes.
If the user asks about a diagram, figure, table, or image, and the retrieved
context flags an element as an image with no extracted text, use the
`read_image` tool on the source file for a visual description.

Only answer from retrieved context and the conversation history. If the
knowledge base has nothing relevant, say so plainly instead of guessing.

Always structure your final response exactly as follows:

Answer:
Provide a clear explanation based on the knowledge base.

Sources:
Mention the documents or files used.

Follow-up:
Suggest one useful follow-up question the user could ask.
"""


class GroqLLMProvider:
    """LLMProvider implementation — wraps ChatGroq. Kept as the default per
    project constraints; any other class exposing get_chat_model() -> a
    BaseChatModel is a drop-in replacement in build_chat_agent() below."""

    def __init__(
        self,
        api_key: Optional[str],
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0,
    ):
        # Deliberately does NOT validate api_key here: this provider is
        # constructed as a request dependency for every conversation route
        # (Backend/deps.py's get_llm_provider), including CRUD endpoints that
        # never touch the LLM — conversation create/list/rename/delete must
        # work without a configured Groq key. The key is only required when
        # get_chat_model() is actually called, i.e. inside ConversationService.ask().
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def get_chat_model(self) -> ChatGroq:
        if not self._api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not configured — set it in .env to use the chat endpoint."
            )
        return ChatGroq(
            model=self._model,
            groq_api_key=self._api_key,
            temperature=self._temperature,
        )


def build_retrieval_tool(store: VectorStore, top_k: int = DEFAULT_TOP_K) -> StructuredTool:
    """
    Builds a `search_knowledge_base` tool bound to ONE conversation's isolated
    VectorStore instance. This closure is what actually enforces chat
    isolation at the agent level: the tool has no way to see any store other
    than the one it was built with, regardless of what the LLM asks for.
    """

    def _search(query: str) -> str:
        results = store.similarity_search_with_score(query, k=top_k)
        if not results:
            return "No relevant information found in this conversation's knowledge base."

        parts = []
        for doc, _score in results:
            meta = doc.metadata
            source = meta.get("research_paper_name", meta.get("file_name", "Unknown"))
            parts.append(f"[Source: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    return StructuredTool.from_function(
        func=_search,
        name="search_knowledge_base",
        description=(
            "Search THIS conversation's uploaded documents for content relevant to a "
            "query. Always call this before answering a question about the user's notes."
        ),
    )


def _trim_history(state: dict) -> dict:
    """pre_model_hook: bounds what's sent to the LLM without touching the
    checkpointer's stored state. Returning `llm_input_messages` (rather than
    `messages`) affects only this model call, not the persisted thread."""
    trimmed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,  # counts messages, not tokens — see MAX_HISTORY_MESSAGES
        max_tokens=MAX_HISTORY_MESSAGES,
        start_on="human",
        include_system=True,
    )
    return {"llm_input_messages": trimmed}


def build_chat_agent(llm_provider, tools: List, checkpointer):
    """
    Compile a LangGraph react-agent wired with the given tools and
    checkpointer. `thread_id` (= chat_id) is supplied by the caller at invoke
    time via config={"configurable": {"thread_id": ...}} — see
    Backend/conversation_service.py — which is what scopes memory to one
    conversation, the same way build_retrieval_tool() scopes documents.
    """
    return create_react_agent(
        model=llm_provider.get_chat_model(),
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        pre_model_hook=_trim_history,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI — manual smoke test. Builds everything locally (no module-level
# singletons); only runs when this file is executed directly.
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    from RAG.vector_store import DEFAULT_PERSIST_DIR
    from Retrieve.checkpointer import InMemoryCheckpointBackend

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY missing in .env")

    demo_store = VectorStore(persist_dir=DEFAULT_PERSIST_DIR, collection="cli_demo")
    demo_llm = GroqLLMProvider(api_key=groq_api_key)
    demo_agent = build_chat_agent(
        llm_provider=demo_llm,
        tools=[build_retrieval_tool(demo_store)],
        checkpointer=InMemoryCheckpointBackend().get_saver(),
    )

    print("\nRagMind CLI demo (in-memory, single session)\n")
    thread_config = {"configurable": {"thread_id": "cli-demo"}}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        result = demo_agent.invoke({"messages": [("user", user_input)]}, config=thread_config)
        print("\n" + "-" * 60)
        print(result["messages"][-1].content)
        print("-" * 60 + "\n")
