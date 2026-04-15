"""Write embedded chunks to the configured vector store."""

from typing import List

from app.domain.entities.embedded_chunk import EmbeddedChunk
from app.domain.ports.vector_store_port import VectorStorePort


class PersistEmbeddedToVectorStoreUseCase:
    """Persists all embedded chunks; returns how many were stored."""

    def __init__(self, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

    def execute(self, chunks: List[EmbeddedChunk]) -> int:
        self._vector_store.replace_all(chunks)
        return len(chunks)
