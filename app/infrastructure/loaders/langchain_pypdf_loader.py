"""PDF loading via LangChain PyPDFLoader (pypdf under the hood)."""

from typing import List

from langchain_community.document_loaders import PyPDFLoader

from app.domain.entities.raw_document import RawDocument
from app.domain.ports.document_loader_port import DocumentLoaderPort


class LangChainPyPDFLoaderAdapter(DocumentLoaderPort):
    """Infrastructure adapter: maps LangChain documents to domain RawDocument."""

    def load_pdf(self, file_path: str) -> List[RawDocument]:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        return [
            RawDocument(page_content=d.page_content, metadata=dict(d.metadata))
            for d in docs
        ]
