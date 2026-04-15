"""Attach embedding vectors to text chunks."""

from typing import List

from app.domain.entities.embedded_chunk import EmbeddedChunk
from app.domain.entities.raw_document import RawDocument
from app.domain.ports.embeddings_port import EmbeddingsPort


class EmbedDocumentsUseCase:
    """Embeds each chunk's text; output order matches input."""

    def __init__(self, embeddings: EmbeddingsPort) -> None:
        self._embeddings = embeddings

    def execute(self, documents: List[RawDocument]) -> List[EmbeddedChunk]:
        if not documents:
            return []
        texts = [d.page_content for d in documents]
        vectors = self._embeddings.embed_texts(texts)
        if len(vectors) != len(documents):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} vectors for {len(documents)} chunks"
            )
        return [
            EmbeddedChunk(
                page_content=doc.page_content,
                metadata=doc.metadata,
                embedding=tuple(vec),
            )
            for doc, vec in zip(documents, vectors, strict=True)
        ]
