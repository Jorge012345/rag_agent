"""Text splitter adapter using LangChain recursive character splitter."""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.domain.entities.raw_document import RawDocument
from app.domain.ports.text_splitter_port import TextSplitterPort


class LangChainRecursiveCharacterSplitterAdapter(TextSplitterPort):
    """Infrastructure adapter for document chunking."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def split_documents(self, documents: List[RawDocument]) -> List[RawDocument]:
        lc_docs = [
            Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in documents
        ]
        chunks = self._splitter.split_documents(lc_docs)
        return [
            RawDocument(page_content=chunk.page_content, metadata=dict(chunk.metadata))
            for chunk in chunks
        ]
