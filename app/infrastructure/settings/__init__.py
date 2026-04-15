"""Aggregated settings (project1-style, env files under app/env/)."""

from pydantic_settings import BaseSettings

from app.infrastructure.settings.database import DatabaseSettings
from app.infrastructure.settings.langchain import LangChainSettings
from app.infrastructure.settings.openai import OpenAISettings
from app.infrastructure.settings.pdf_paths import PdfPathsSettings


class Settings(BaseSettings):
    """Global settings composed from nested env files."""

    OPENAI: OpenAISettings
    DATABASE: DatabaseSettings
    LANGCHAIN: LangChainSettings
    PDF_PATHS: PdfPathsSettings


SETTINGS = Settings(
    OPENAI=OpenAISettings(),
    DATABASE=DatabaseSettings(),
    LANGCHAIN=LangChainSettings(),
    PDF_PATHS=PdfPathsSettings(),
)

__all__ = [
    "SETTINGS",
    "Settings",
    "OpenAISettings",
    "DatabaseSettings",
    "LangChainSettings",
    "PdfPathsSettings",
]
