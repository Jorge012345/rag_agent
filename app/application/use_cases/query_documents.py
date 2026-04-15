"""Semantic search over ingested chunks in the vector store."""

from typing import List, Optional

from app.domain.entities.search_hit import SearchHit
from app.domain.ports.embeddings_port import EmbeddingsPort
from app.domain.ports.vector_search_port import VectorSearchPort


class QueryDocumentsUseCase:
    """Embeds the user question and runs similarity search in Chroma."""

    def __init__(
        self,
        embeddings: EmbeddingsPort,
        vector_search: VectorSearchPort,
        default_k: int,
    ) -> None:
        self._embeddings = embeddings
        self._vector_search = vector_search
        self._default_k = default_k

    def execute(self, query: str, k: Optional[int] = None) -> List[SearchHit]:
        text = query.strip()
        if not text:
            raise ValueError("Query text must not be empty.")
        top_k = k if k is not None else self._default_k
        if top_k < 1:
            raise ValueError("k must be at least 1.")
        vector = self._embeddings.embed_query(text)
        return self._vector_search.search_by_vector(vector, top_k)
