"""HTTP request/response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LoadDocumentsRequest(BaseModel):
    """Paths to PDF files (project root–relative or absolute)."""

    paths: Optional[List[str]] = Field(
        None,
        description="PDF paths. Omit or null to use paths from app/env/pdf_paths.env (PDF_PATHS).",
    )


class LoadedChunkOut(BaseModel):
    """One loaded text chunk (preview only; full text can be large)."""

    metadata: Dict[str, Any]
    content_length: int = Field(description="Full text length in characters")
    content_preview: str = Field(
        max_length=500,
        description="Start of chunk text (truncated for response size)",
    )


class LoadDocumentsResponse(BaseModel):
    """Load PDFs → split into chunks → embed (OpenAI). Full vectors are not returned."""

    documents_count: int = Field(description="Number of chunks after split")
    sources: List[str] = Field(description="PDF file paths that were read")
    preview_limit: int = Field(
        description="Max number of chunk previews returned in `documents`",
    )
    documents: List[LoadedChunkOut] = Field(
        description="Previews for the first N chunks (see preview_limit)",
    )
    embedding_dimensions: int = Field(
        description="Length of each embedding vector produced for chunks",
    )
    first_vector_preview: List[float] = Field(
        description="First N dimensions of the first chunk's embedding (preview only)",
    )
    chunks_stored_in_chroma: int = Field(
        description="Number of embedded chunks written to ChromaDB",
    )
    chroma_collection_name: str
    chroma_persist_directory: str = Field(
        description="On-disk path for the Chroma persistent client",
    )


class QueryDocumentsRequest(BaseModel):
    """Natural-language question over ingested PDF chunks."""

    query: str = Field(min_length=1, description="Question or search text")
    k: Optional[int] = Field(
        None,
        ge=1,
        description="Max chunks to return; defaults to RETRIEVER_K from langchain.env",
    )


class QueryHitOut(BaseModel):
    """One similar chunk from Chroma."""

    content: str
    metadata: Dict[str, Any]
    distance: Optional[float] = Field(
        None,
        description="Chroma distance for this hit (lower is usually more similar for cosine space)",
    )


class QueryDocumentsResponse(BaseModel):
    """Semantic search results."""

    query: str
    results_count: int
    results: List[QueryHitOut]


class AskDocumentsRequest(BaseModel):
    """Question answered from retrieved context (RAG)."""

    query: str = Field(min_length=1, description="Question about ingested documents")
    k: Optional[int] = Field(
        None,
        ge=1,
        description="Chunks to pass as context; defaults to RETRIEVER_K from langchain.env",
    )


class AskDocumentsResponse(BaseModel):
    """Natural-language answer grounded in similar chunks."""

    query: str
    answer: str
    context_chunks_used: int
    context: List[QueryHitOut] = Field(
        description="Retrieved chunks the model was given (verbatim context)",
    )
