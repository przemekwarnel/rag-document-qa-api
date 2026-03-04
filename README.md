# RAG Bot – Document Question Answering API

This project implements a lightweight Retrieval-Augmented Generation (RAG) system exposed as a REST API using FastAPI.

The system allows users to:

- Upload a PDF document
- Index its content into a vector database (Chroma)
- Ask questions about the document
- Receive grounded answers with supporting source snippets

The architecture combines:

- Dense vector retrieval (SentenceTransformers + Chroma)
- Context-constrained LLM generation (FLAN-T5)
- Deterministic lexical verification for "mention/contain/include" queries

The API is fully containerized with Docker and designed to be reproducible and easy to deploy.

## Architecture & System Flow

The system follows a standard Retrieval-Augmented Generation (RAG) pipeline:

```
User query
      │
      ▼
FastAPI `/ask`
      │
      ▼
Retriever (Chroma)
      │
      ▼
Top-k document chunks
      │
      ▼
Context construction
      │
      ▼
LLM (FLAN-T5)
      │
      ▼
Answer + sources
```

### 1. Ingestion Phase

- A PDF file is uploaded via `/ingest`
- The document is split into chunks
- Each chunk is embedded using SentenceTransformers (`all-MiniLM-L6-v2`)
- Embeddings are stored in a Chroma vector database

### 2. Retrieval Phase

When a query is sent to `/ask`:

- The query is embedded using the same embedding model
- Top-k most similar chunks are retrieved from Chroma
- Retrieved chunks are treated as the grounding context

### 3. Answer Generation

Two answer modes exist:

#### A) Lexical Verification Mode
If the query contains keywords such as:
- "mention"
- "contain"
- "include"
- "appear"
- "occur"

The system performs deterministic lexical matching on retrieved chunks.
This avoids unnecessary LLM hallucination for existence-based queries.

#### B) Generative QA Mode
For all other queries:

- Retrieved chunks are concatenated (up to `max_context_chars`)
- A constrained prompt is built:
  > "Answer using ONLY the context below"
- FLAN-T5 generates the final answer
- Supporting snippets are returned for traceability

---

### Design Principles

- No black-box RAG chains
- Explicit retrieval → context construction → generation
- Transparent source attribution
- Dockerized and reproducible

## Tech Stack

- **Python**
- **FastAPI** — REST API
- **SentenceTransformers** — embedding generation
- **Chroma** — vector database
- **Transformers (FLAN-T5)** — answer generation
- **Docker** — containerization

## Project Structure

```
rag-bot/
│
├── app.py                # FastAPI application and API endpoints
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
├── README.md
│
├── rag/
│   ├── pipeline.py       # RAG pipeline (retrieval + generation logic)
│   └── ingestion.py      # Document loading, chunking, and indexing
│
├── uploads/              # Uploaded documents
├── chroma_index/         # Persistent vector database
│
├── test_pipeline.py      # Simple pipeline test
└── test_ingestion.py     # Indexing test
```

### Key Components

- **FastAPI API layer** (`app.py`)  
  Handles document ingestion and question answering endpoints.

- **RAG pipeline** (`rag/pipeline.py`)  
  Implements retrieval, context construction, and LLM-based answer generation.

- **Document ingestion** (`rag/ingestion.py`)  
  Splits PDF documents into chunks, generates embeddings, and stores them in Chroma.

- **Vector database** (`chroma_index/`)  
  Persistent storage for document embeddings used during retrieval.

## API Endpoints

### `POST /ingest`

Uploads and indexes a PDF document.

**Request (multipart/form-data):**
- `file`: PDF document

**Behavior:**
- Saves file to `/uploads`
- Builds a new Chroma index
- Reloads the RAG pipeline

**Response:**
```json
{
  "status": "ok",
  "filename": "example.pdf"
}
```

---

### `POST /ask`

Answers a question based on the indexed document.

**Request (application/json):**
```json
{
  "query": "What is self-attention?"
}
```

**Response:**
```json
{
  "answer": "An attention mechanism relating different positions...",
  "sources": [
    {
      "snippet": "...",
      "metadata": { ... }
    }
  ]
}
```

---

### Response Structure
- answer - generated or deterministic response 
- sources - top supporting chunks for transparency and traceability 

## How to Run

### Option A — Run Locally

#### 1. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Start the API server
```bash
uvicorn app:app --reload
```

#### Swagger UI will be available at:
```
http://127.0.0.1:8000/docs
```

---

### Option B - Run with Docker

#### 1. Build image
```bash
docker build -t rag-bot .
```

#### 2. Run container
```bash
docker run -p 8000:8000 rag-bot
```

#### Swagger UI
```
http://127.0.0.1:8000/docs
```

---

### Example Usage (CLI)

#### Ingest a document:
```bash
curl -X POST "http://127.0.0.1:8000/ingest" \
  -F "file=@test.pdf"
  ```

#### Ask a question:
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is self-attention?"}'
```

## Limitations & Future Improvements

### Current Limitations

- Retrieval is limited to top-k dense similarity search.
- "No" answers in mention-based queries mean:
  > "not found in top-k retrieved chunks"
  and do not guarantee absence in the full document.
- Context size is limited by `max_context_chars`.
- The system indexes one document at a time (new ingest replaces the previous index).

---

### Possible Improvements

- Hybrid retrieval (dense + lexical search)  
  Combining semantic embeddings with a lexical method such as BM25 would improve recall for rare or domain-specific terms.

- Cross-encoder reranking  
  Retrieved chunks could be reranked using a cross-encoder model to improve precision before passing context to the LLM.

- Multi-document indexing  
  Instead of replacing the index on each ingest, the system could support multiple documents with metadata-based filtering.

---

### Design Tradeoff

The project intentionally favors:

- Explicit, understandable RAG flow
- Minimal abstraction layers
- Transparency over complexity