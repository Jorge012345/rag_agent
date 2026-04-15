"""Composition root: wire ports to adapters and use cases."""

from fastapi import Depends

from app.application.use_cases.ask_documents import AskDocumentsUseCase
from app.application.use_cases.embed_documents import EmbedDocumentsUseCase
from app.application.use_cases.load_pdfs import LoadPdfsUseCase
from app.application.use_cases.persist_to_vector_store import (
    PersistEmbeddedToVectorStoreUseCase,
)
from app.application.use_cases.query_documents import QueryDocumentsUseCase
from app.infrastructure.embeddings.langchain_openai_embeddings import (
    LangChainOpenAIEmbeddingsAdapter,
)
from app.infrastructure.llm.langchain_openai_chat import LangChainOpenAIChatAdapter
from app.infrastructure.loaders.langchain_pypdf_loader import LangChainPyPDFLoaderAdapter
from app.infrastructure.splitters.langchain_recursive_character_splitter import (
    LangChainRecursiveCharacterSplitterAdapter,
)
from app.infrastructure.vectorstores.chroma_vector_search import ChromaVectorSearchAdapter
from app.infrastructure.vectorstores.chroma_vector_store import ChromaVectorStoreAdapter
from app.infrastructure.settings import SETTINGS


def get_load_pdfs_use_case() -> LoadPdfsUseCase:
    return LoadPdfsUseCase(
        document_loader=LangChainPyPDFLoaderAdapter(),
        text_splitter=LangChainRecursiveCharacterSplitterAdapter(
            chunk_size=SETTINGS.LANGCHAIN.CHUNK_SIZE,
            chunk_overlap=SETTINGS.LANGCHAIN.CHUNK_OVERLAP,
        ),
        default_pdf_paths=SETTINGS.PDF_PATHS.pdf_path_list(),
    )


def get_embed_documents_use_case() -> EmbedDocumentsUseCase:
    return EmbedDocumentsUseCase(embeddings=LangChainOpenAIEmbeddingsAdapter())


def get_persist_to_vector_store_use_case() -> PersistEmbeddedToVectorStoreUseCase:
    return PersistEmbeddedToVectorStoreUseCase(
        vector_store=ChromaVectorStoreAdapter(),
    )


def get_query_documents_use_case() -> QueryDocumentsUseCase:
    return QueryDocumentsUseCase(
        embeddings=LangChainOpenAIEmbeddingsAdapter(),
        vector_search=ChromaVectorSearchAdapter(),
        default_k=SETTINGS.LANGCHAIN.RETRIEVER_K,
    )


def get_ask_documents_use_case(
    query_uc: QueryDocumentsUseCase = Depends(get_query_documents_use_case),
) -> AskDocumentsUseCase:
    return AskDocumentsUseCase(
        query_documents=query_uc,
        llm=LangChainOpenAIChatAdapter(),
        default_context_k=SETTINGS.LANGCHAIN.RETRIEVER_K,
    )
