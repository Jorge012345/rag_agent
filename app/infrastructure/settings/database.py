"""Database / Chroma settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """ChromaDB configuration."""

    CHROMA_PERSIST_DIRECTORY: str
    CHROMA_COLLECTION_NAME: str

    model_config = SettingsConfigDict(
        env_file="app/env/database.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
