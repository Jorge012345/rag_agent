# Project2

## Configuration (same idea as project1)

Env files under `app/env/`:

| File | Purpose |
|------|---------|
| `openai.env` | API key + model / embeddings (**not in git** — copy from example) |
| `database.env` | Chroma persist path + collection name |
| `langchain.env` | Retriever options + chunking (`CHUNK_SIZE`, `CHUNK_OVERLAP`) |
| `pdf_paths.env` | Comma-separated PDF paths (`PDF_PATHS`) |

## PDFs (clean architecture)

| Layer | Role |
|-------|------|
| **`app/resources/pdfs/`** | Static PDF files (same as project1; copied here so project2 is self-contained). |
| **`app/domain/`** | `RawDocument` + `DocumentLoaderPort` (no LangChain). |
| **`app/application/`** | `LoadPdfsUseCase`, `EmbedDocumentsUseCase`, `PersistEmbeddedToVectorStoreUseCase`. |
| **`app/infrastructure/loaders/`** | `LangChainPyPDFLoaderAdapter` → LangChain `PyPDFLoader` (uses **pypdf**). |
| **`app/infrastructure/splitters/`** | `LangChainRecursiveCharacterSplitterAdapter` (chunking). |
| **`app/infrastructure/embeddings/`** | `LangChainOpenAIEmbeddingsAdapter` → OpenAI embeddings. |
| **`app/infrastructure/vectorstores/`** | `ChromaVectorStoreAdapter` → persisted ChromaDB. |
| **`app/interfaces/api/v1/`** | HTTP routes + DI wiring. |

To use project1 paths only (no copy), set absolute paths in `app/env/pdf_paths.env`, e.g.  
`PDF_PATHS=/path/to/project1/app/utilities/pdfs/libro.pdf,...`

Setup OpenAI:

```bash
cp app/env/openai.env.example app/env/openai.env
# Edit app/env/openai.env and set OPENAI_API_KEY
```

Run from **project2** root so paths like `app/env/...` resolve correctly:

```bash
uv sync
uv run uvicorn app.main:app --reload
```

**Ingest pipeline:** `POST /api/v1/documents/load`  
Body `{}` uses `PDF_PATHS` from `pdf_paths.env`, or `{ "paths": ["app/resources/pdfs/libro.pdf"] }`.  
Flow: **load PDFs → split → embed (OpenAI) → store all vectors in Chroma** (`database.env`: `CHROMA_PERSIST_DIRECTORY`, `CHROMA_COLLECTION_NAME`). Each run **replaces** the collection contents (full re-ingest).  
Requires valid `OPENAI_API_KEY` in `app/env/openai.env`. Response includes chunk previews, embedding preview fields, and `chunks_stored_in_chroma` (full vectors are not returned in JSON).

**Query ingested content:** `POST /api/v1/documents/query`  
Body `{ "query": "your question" }`, optional `"k": 4` (defaults to `RETRIEVER_K` in `langchain.env`).  
Requires a prior successful `/documents/load` and a valid API key for query embeddings. Returns matching chunk `content`, `metadata`, and `distance`.

Use settings in code:

```python
from app.infrastructure.settings import SETTINGS

SETTINGS.OPENAI.OPENAI_API_KEY
SETTINGS.DATABASE.CHROMA_PERSIST_DIRECTORY
SETTINGS.LANGCHAIN.RETRIEVER_K
SETTINGS.PDF_PATHS.pdf_path_list()
```
