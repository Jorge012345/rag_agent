"""OpenAI embeddings via LangChain (infrastructure)."""

from typing import List, Sequence

from langchain_openai import OpenAIEmbeddings

from app.domain.ports.embeddings_port import EmbeddingsPort
from app.infrastructure.settings import SETTINGS


class LangChainOpenAIEmbeddingsAdapter(EmbeddingsPort):
    """Maps texts to vectors using OpenAI embedding models."""

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            openai_api_key=SETTINGS.OPENAI.OPENAI_API_KEY,
            model=SETTINGS.OPENAI.EMBEDDING_ENGINE,
            dimensions=SETTINGS.OPENAI.EMBEDDING_DIMENSIONS,
        )

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embeddings.embed_documents(list(texts))

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)
