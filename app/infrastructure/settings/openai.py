"""OpenAI settings (same pattern as project1)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """OpenAI configuration."""

    OPENAI_API_KEY: str
    EMBEDDING_ENGINE: str
    EMBEDDING_DIMENSIONS: int
    MODEL_NAME: str
    TEMPERATURE: float

    model_config = SettingsConfigDict(
        env_file="app/env/openai.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
