"""Port: similarity search over stored vectors."""

from abc import ABC, abstractmethod
from typing import List, Sequence

from app.domain.entities.search_hit import SearchHit


class VectorSearchPort(ABC):
    """Query the vector database by embedding vector."""

    @abstractmethod
    def search_by_vector(
        self,
        query_embedding: Sequence[float],
        k: int,
    ) -> List[SearchHit]:
        """Return up to k most similar stored chunks."""
