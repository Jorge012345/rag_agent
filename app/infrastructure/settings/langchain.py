"""LangChain retriever settings."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class LangChainSettings(BaseSettings):
    """LangChain retriever configuration."""

    RETRIEVER_K: int = 4
    RETRIEVER_SEARCH_TYPE: str = "similarity"
    RETRIEVER_SCORE_THRESHOLD: Optional[float] = None
    RETRIEVER_FETCH_K: Optional[int] = None
    RETRIEVER_LAMBDA_MULT: Optional[float] = None
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200

    model_config = SettingsConfigDict(
        env_file="app/env/langchain.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
