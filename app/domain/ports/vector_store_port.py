"""Port: persist embedded chunks in a vector database."""

from abc import ABC, abstractmethod
from typing import Sequence

from app.domain.entities.embedded_chunk import EmbeddedChunk


class VectorStorePort(ABC):
    """Abstraction over vector DB (Chroma, etc.)."""

    @abstractmethod
    def replace_all(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Replace collection contents with these chunks (full re-ingest)."""
