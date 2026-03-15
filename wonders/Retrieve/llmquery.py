"""
llmquery.py
===========

Direct RAG pipeline.

User Query
    │
    ▼
Search KB  ----> Vector DB
    │
    ▼
Context
    │
    ▼
LLM generates answer
    │
    ▼
Final answer
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Ensure folders are importable
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent

RAG_DIR = BASE_DIR / "RAG"

sys.path.insert(0, str(RAG_DIR))

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vector_store import VectorStore, DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION

# ─────────────────────────────────────────────
# Load environment variables
# ─────────────────────────────────────────────

load_dotenv(BASE_DIR / ".env")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

TOP_K = 5

# ─────────────────────────────────────────────
# Initialize vector store
# ─────────────────────────────────────────────

_store = VectorStore(
    persist_dir=DEFAULT_PERSIST_DIR,
    collection=DEFAULT_COLLECTION,
)

# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────


def search_knowledge_base(query: str):
    """
    Search the vector database for relevant chunks.
    """

    results = _store.similarity_search_with_score(query, k=TOP_K)

    if not results:
        return "No relevant information found in the knowledge base.", []

    print(f"\n[RAG] Retrieved {len(results)} chunks\n")

    context_parts = []
    docs = []

    for i, (doc, score) in enumerate(results, 1):

        meta = doc.metadata

        source = meta.get("research_paper_name", meta.get("source", "Unknown"))
        chunk = f"{meta.get('chunk_index','?')}/{meta.get('chunk_total','?')}"

        print(f"[{i}] score={score:.4f} | {source} | chunk {chunk}")
        print(f"     {doc.page_content[:150]}\n")

        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
        docs.append(doc)

    return "\n\n---\n\n".join(context_parts), docs


# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

_SYSTEM_PROMPT = """
You are a helpful document assistant.
You must use the provided context documents to help answer user queries.
Just focus on answering the user's request using the retrieved documents.

Context Information:
{context}

Always structure your final response exactly as follows:

Answer:
Provide a clear explanation based on the knowledge base.

Sources:
Mention the documents or files used.

Follow-up:
Suggest one useful follow-up question the user could ask.
"""

# ─────────────────────────────────────────────
# Setup LLM
# ─────────────────────────────────────────────

_GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not _GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY missing in .env")

_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=_GROQ_API_KEY,
    temperature=0,
)

# ─────────────────────────────────────────────
# Create Pipeline (for compat with backend.py)
# ─────────────────────────────────────────────


class SimpleRAGPipeline:
    def invoke(self, inputs: dict):
        messages_input = inputs.get("messages", [])
        if not messages_input:
            raise ValueError("No messages provided")

        # Extract the user's query
        last_msg = messages_input[-1]
        query = (
            last_msg[1]
            if isinstance(last_msg, tuple)
            else getattr(last_msg, "content", str(last_msg))
        )

        # Retrieve context
        context_str, docs = search_knowledge_base(query)

        # Format prompt
        prompt = _SYSTEM_PROMPT.format(context=context_str)

        messages = [("system", prompt), ("user", query)]

        response = _llm.invoke(messages)

        return {"messages": [response], "chunks": docs}


pipeline = SimpleRAGPipeline()

# ─────────────────────────────────────────────
# Query Function
# ─────────────────────────────────────────────


def ask(query: str) -> str:

    print("\n" + "=" * 60)
    print(f"User Query: {query}")
    print("=" * 60)

    try:

        result = pipeline.invoke({"messages": [("user", query)]})

        answer = result["messages"][-1].content

    except Exception as e:

        print("Error:", e)
        return str(e)

    print("\n" + "-" * 60)
    print("Agent Answer:\n")
    print(answer)
    print("-" * 60 + "\n")

    return answer


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\nMultimodal RAG Assistant\n")

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

        ask(user_input)
