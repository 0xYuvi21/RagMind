# RagMind — Multimodal RAG-Powered Document Q&A System

## Project Summary (Resume-Ready)

**RagMind** is a full-stack, multimodal Retrieval-Augmented Generation (RAG) system that enables users to upload documents in virtually any format and ask natural-language questions, receiving context-grounded answers powered by an LLM.

---

## XYZ Resume Bullets

### 1. Full-Stack RAG Pipeline Architecture
> **X:** Built an end-to-end Retrieval-Augmented Generation (RAG) system with a modular 4-stage pipeline (Ingest → Chunk → Embed → Retrieve)
> **Y:** Supporting 30+ file formats including PDFs, Word docs, spreadsheets, presentations, images (with OCR), e-books, and plain text — enabling multimodal document intelligence
> **Z:** By designing a decoupled architecture using LangChain's UnstructuredLoader for format-agnostic ingestion, RecursiveCharacterTextSplitter for semantic chunking with MD5-based deduplication, HuggingFace `all-MiniLM-L6-v2` embeddings, and ChromaDB as the persistent vector store

### 2. Intelligent Document Ingestion & Processing
> **X:** Engineered a multimodal document ingestion engine capable of processing both static files and dynamic in-memory uploads
> **Y:** Automatically extracting, filtering, and enriching content from documents, spreadsheets, images (via OCR for scanned/handwritten content), presentations, and emails — retaining only semantically meaningful elements (NarrativeText, Tables, Formulas, FigureCaptions)
> **Z:** By implementing a [DocumentIngestor](file:///c:/Users/muthi/Desktop/wonders/RAG/ingest.py#72-302) class with Unstructured API integration (supporting both local and remote partitioning), element-category filtering to discard headers/footers/boilerplate, and enriched metadata generation (source tracking, word counts, content previews, research paper name extraction)

### 3. Vector Store & Semantic Retrieval System
> **X:** Developed a persistent vector database layer with chunk-level deduplication and cosine-similarity search
> **Y:** Enabling sub-second semantic retrieval of the top-k most relevant document chunks across an ever-growing knowledge base, with zero duplicate vectors even upon re-ingestion
> **Z:** By building a [VectorStore](file:///c:/Users/muthi/Desktop/wonders/RAG/vector_store.py#61-301) class on top of ChromaDB with HuggingFace sentence-transformer embeddings, MD5-based `chunk_id` upserts for idempotent storage, complex metadata filtering, and a LangChain-compatible [as_retriever()](file:///c:/Users/muthi/Desktop/wonders/RAG/vector_store.py#240-252) interface for seamless chain/agent integration

### 4. LLM-Powered Answer Generation with Source Attribution
> **X:** Integrated a RAG retrieval pipeline with Groq's `llama-3.3-70b-versatile` LLM for context-grounded answer generation
> **Y:** Producing structured responses with clear answers, cited sources, and auto-generated follow-up questions — reducing LLM hallucinations by anchoring responses to retrieved document chunks
> **Z:** By implementing a [SimpleRAGPipeline](file:///c:/Users/muthi/Desktop/wonders/Retrieve/llmquery.py#148-173) that retrieves top-5 relevant chunks via vector similarity search, formats them as context in a system prompt, and invokes the LLM through LangChain's ChatGroq integration with temperature=0 for deterministic, factual responses

### 5. FastAPI Backend with Dynamic File Upload & Chat API
> **X:** Built a RESTful API backend serving the RAG pipeline with real-time file ingestion capabilities
> **Y:** Allowing users to upload documents on-the-fly via multipart form data, dynamically expanding the knowledge base mid-conversation, and returning answers with parsed source attribution
> **Z:** By developing a FastAPI server with CORS-enabled endpoints, in-memory file stream ingestion ([add_file_stream](file:///c:/Users/muthi/Desktop/wonders/RAG/vector_store.py#145-175)), automatic source extraction from both metadata and LLM response text via regex parsing, and graceful error handling with user-friendly fallback messages

### 6. Polished Conversational Chat UI
> **X:** Designed and developed a responsive, single-page chat interface with a modern, premium aesthetic
> **Y:** Featuring drag-and-drop file uploads, word-by-word response animations, clickable follow-up questions, markdown rendering, source citation chips, multi-session chat management, and a sidebar with conversation history
> **Z:** By building a 1,480-line vanilla HTML/CSS/JS frontend with Google Fonts (DM Sans + Lora), CSS custom properties for theming, the `marked.js` library for markdown parsing, a custom DOM-walking word-fade animation engine, and drag-and-drop event handling with visual overlays and toast notifications

---

## Supported File Formats (30+)

| Category         | Formats                                                       |
|------------------|---------------------------------------------------------------|
| **Documents**    | PDF, DOC, DOCX, ODT, RTF                                     |
| **Spreadsheets** | XLS, XLSX, ODS, CSV, TSV                                     |
| **Presentations**| PPT, PPTX                                                    |
| **Text/Markup**  | TXT, MD, RST, HTML, XML, JSON                                |
| **Email**        | EML, MSG                                                      |
| **Images (OCR)** | JPG, JPEG, PNG, BMP, TIFF, TIF, GIF, WEBP                   |
| **E-books**      | EPUB                                                          |

---

## Tech Stack

| Layer       | Technology                                                                          |
|-------------|-------------------------------------------------------------------------------------|
| **Frontend**| HTML5, CSS3, Vanilla JS, marked.js, Google Fonts                                    |
| **Backend** | FastAPI, Uvicorn, Python 3.13                                                       |
| **LLM**     | Groq API (Llama 3.3 70B Versatile), LangChain, LangGraph                           |
| **RAG**     | LangChain UnstructuredLoader, RecursiveCharacterTextSplitter                        |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` (sentence-transformers)                          |
| **Vector DB**  | ChromaDB (persistent, on-disk)                                                   |
| **Tooling** | uv (package manager), python-dotenv                                                 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (index.html)                       │
│  Chat UI · Drag-and-Drop · Markdown · Follow-ups · Sessions    │
└────────────────────────────┬────────────────────────────────────┘
                             │ POST /api/chat (FormData)
┌────────────────────────────▼────────────────────────────────────┐
│                   Backend (FastAPI)                              │
│          File Upload Handling · Source Attribution               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              Retrieve (llmquery.py)                              │
│     RAG Pipeline: Vector Search → Context → LLM → Answer       │
│               Groq Llama 3.3 70B · LangChain                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                    RAG Pipeline                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐               │
│  │ ingest.py│──▶│chunking  │──▶│ vector_store  │               │
│  │ 30+ fmts │   │ .py      │   │ .py           │               │
│  │ OCR/API  │   │ Dedup    │   │ ChromaDB      │               │
│  └──────────┘   │ Overlap  │   │ HuggingFace   │               │
│                 └──────────┘   │ Embeddings    │               │
│                                └──────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```
