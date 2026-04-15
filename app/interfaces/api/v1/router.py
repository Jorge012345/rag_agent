"""V1 routes (thin: delegate to use cases)."""

import os

from fastapi import APIRouter, Depends, HTTPException

from app.application.use_cases.ask_documents import AskDocumentsUseCase
from app.application.use_cases.embed_documents import EmbedDocumentsUseCase
from app.application.use_cases.load_pdfs import LoadPdfsUseCase
from app.application.use_cases.persist_to_vector_store import (
    PersistEmbeddedToVectorStoreUseCase,
)
from app.application.use_cases.query_documents import QueryDocumentsUseCase
from app.infrastructure.settings import SETTINGS
from app.interfaces.api.v1.dependencies import (
    get_ask_documents_use_case,
    get_embed_documents_use_case,
    get_load_pdfs_use_case,
    get_persist_to_vector_store_use_case,
    get_query_documents_use_case,
)
from app.interfaces.api.v1.schemas import (
    AskDocumentsRequest,
    AskDocumentsResponse,
    LoadDocumentsRequest,
    LoadDocumentsResponse,
    LoadedChunkOut,
    QueryDocumentsRequest,
    QueryDocumentsResponse,
    QueryHitOut,
)

router = APIRouter(prefix="/api/v1")

_PREVIEW_MAX_CHUNKS = 100
_PREVIEW_CHARS = 400
_EMBED_VECTOR_PREVIEW_DIMS = 16


def _resolve_pdf_paths(body: LoadDocumentsRequest) -> list[str]:
    if body.paths is not None and len(body.paths) == 0:
        raise HTTPException(
            status_code=400,
            detail="paths cannot be empty; omit the field to use PDF_PATHS from configuration.",
        )
    paths_used = (
        list(body.paths)
        if body.paths is not None
        else SETTINGS.PDF_PATHS.pdf_path_list()
    )
    if not paths_used:
        raise HTTPException(
            status_code=400,
            detail="No PDF paths configured. Set paths in the body or PDF_PATHS in app/env/pdf_paths.env.",
        )
    return paths_used


@router.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/documents/load",
    response_model=LoadDocumentsResponse,
    summary="Load PDFs, split into chunks, and embed",
    tags=["documents"],
)
def load_documents(
    body: LoadDocumentsRequest,
    load_uc: LoadPdfsUseCase = Depends(get_load_pdfs_use_case),
    embed_uc: EmbedDocumentsUseCase = Depends(get_embed_documents_use_case),
    persist_uc: PersistEmbeddedToVectorStoreUseCase = Depends(
        get_persist_to_vector_store_use_case
    ),
) -> LoadDocumentsResponse:
    """PDF → chunks → embeddings → ChromaDB (response includes previews only, not full vectors)."""
    paths_used = _resolve_pdf_paths(body)
    chunks = load_uc.execute(pdf_paths=paths_used)
    embedded = embed_uc.execute(chunks)
    stored = persist_uc.execute(embedded)

    dims = (
        len(embedded[0].embedding)
        if embedded
        else SETTINGS.OPENAI.EMBEDDING_DIMENSIONS
    )
    head = (
        list(embedded[0].embedding[:_EMBED_VECTOR_PREVIEW_DIMS])
        if embedded
        else []
    )

    previews: list[LoadedChunkOut] = []
    for doc in chunks[:_PREVIEW_MAX_CHUNKS]:
        text = doc.page_content
        previews.append(
            LoadedChunkOut(
                metadata=dict(doc.metadata),
                content_length=len(text),
                content_preview=text[:_PREVIEW_CHARS]
                + ("…" if len(text) > _PREVIEW_CHARS else ""),
            )
        )

    persist_dir = SETTINGS.DATABASE.CHROMA_PERSIST_DIRECTORY
    persist_abs = persist_dir if os.path.isabs(persist_dir) else os.path.abspath(persist_dir)

    return LoadDocumentsResponse(
        documents_count=len(chunks),
        sources=paths_used,
        preview_limit=_PREVIEW_MAX_CHUNKS,
        documents=previews,
        embedding_dimensions=dims,
        first_vector_preview=head,
        chunks_stored_in_chroma=stored,
        chroma_collection_name=SETTINGS.DATABASE.CHROMA_COLLECTION_NAME,
        chroma_persist_directory=persist_abs,
    )


@router.post(
    "/documents/query",
    response_model=QueryDocumentsResponse,
    summary="Query ingested documents (semantic search in Chroma)",
    tags=["documents"],
)
def query_documents(
    body: QueryDocumentsRequest,
    query_uc: QueryDocumentsUseCase = Depends(get_query_documents_use_case),
) -> QueryDocumentsResponse:
    """Requires prior `POST /documents/load`. Embeds `query` and returns similar chunks."""
    try:
        hits = query_uc.execute(body.query, body.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return QueryDocumentsResponse(
        query=body.query.strip(),
        results_count=len(hits),
        results=[
            QueryHitOut(
                content=h.page_content,
                metadata=dict(h.metadata),
                distance=h.distance,
            )
            for h in hits
        ],
    )


@router.post(
    "/documents/ask",
    response_model=AskDocumentsResponse,
    summary="Ask about ingested documents (RAG: retrieve + LLM answer)",
    tags=["documents"],
)
def ask_documents(
    body: AskDocumentsRequest,
    ask_uc: AskDocumentsUseCase = Depends(get_ask_documents_use_case),
) -> AskDocumentsResponse:
    """Retrieves top-k chunks, then uses the configured chat model to answer in natural language."""
    try:
        result = ask_uc.execute(body.query, body.k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    hits = result.context_hits
    return AskDocumentsResponse(
        query=body.query.strip(),
        answer=result.answer,
        context_chunks_used=len(hits),
        context=[
            QueryHitOut(
                content=h.page_content,
                metadata=dict(h.metadata),
                distance=h.distance,
            )
            for h in hits
        ],
    )
