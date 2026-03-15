"""
FastAPI backend for the RAG pipeline.
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add project roots to path so we can import RAG modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Retrieve"))
sys.path.insert(0, str(PROJECT_ROOT / "RAG"))

import os
import re
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure uploads directory exists (use tempfile to prevent VS Code Live Server from auto-reloading the page)
UPLOADS_DIR = Path(tempfile.gettempdir()) / "ragmind_uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Import our pipeline and vector store
from llmquery import pipeline, _store

app = FastAPI(title="RagMind API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
async def chat(query: str = Form(...), files: Optional[List[UploadFile]] = File(None)):
    """
    Process a chat message, optionally ingesting new files first.
    """
    # 1. Ingest any uploaded files dynamically
    saved_paths = []
    if files:
        for f in files:
            content = await f.read()
            if content:
                print(f"Ingesting uploaded file: {f.filename}")

                # Save physically to disk for VLM/Tools
                safe_name = f.filename.replace(" ", "_")
                disk_path = UPLOADS_DIR / safe_name
                with open(disk_path, "wb") as out_f:
                    out_f.write(content)

                saved_paths.append(str(disk_path.resolve()))

                try:
                    _store.add_file_stream(
                        file_content=content,
                        file_name=f.filename,
                        source_tag="user_upload",
                    )
                except Exception as e:
                    print(f"Skipping {f.filename}: {e}")

    # Tell the Agent about the files available in the vector store
    if saved_paths:
        file_list = "\n".join([f"- {Path(p).name}" for p in saved_paths])
        query = f"{query}\n\n[System Note: The following files were just uploaded and added to the knowledge base:\n{file_list}]"

    # 2. Query the Agent pipeline (using standard messages)
    try:
        result = pipeline.invoke({"messages": [("user", query)]})
        # The LangGraph react agent returns the final message in the 'messages' array
        answer = result["messages"][-1].content
    except Exception as e:
        print(f"Agent Error: {e}")
        return {
            "text": "Sorry, I encountered an error while processing your request. Please try again or rephrase your query.",
            "sources": [],
        }

    # 3. Extract unique sources to send back to the frontend
    sources = set()
    for doc in result.get("chunks", []):
        m = doc.metadata
        # Get the research paper name if available, fallback to the file name or source
        src = m.get(
            "research_paper_name", m.get("file_name", m.get("source", "Unknown"))
        )
        sources.add(src)

    # Any newly uploaded files in this request are immediately added to context, so they count as sources
    if files:
        for f in files:
            sources.add(f.filename)

    # Finally, also explicitly add any sources the LLM mentioned in the "Sources:" block
    sources_match = re.search(
        r"Sources:\s*(.*?)(?:\n\nFollow-up:|$)", answer, re.IGNORECASE | re.DOTALL
    )
    if sources_match:
        text_sources = sources_match.group(1).split("\n")
        for raw_src in text_sources:
            clean_src = raw_src.strip("-* \t")
            if clean_src:
                sources.add(clean_src)

    return {"text": answer, "sources": list(sources)}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)
