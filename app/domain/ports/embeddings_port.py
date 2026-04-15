"""Port: turn text into embedding vectors."""

from abc import ABC, abstractmethod
from typing import List, Sequence


class EmbeddingsPort(ABC):
    """Abstraction over an embedding model (OpenAI, local, etc.)."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Return one vector per input text (same order, same length)."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (same model as document embeddings)."""
