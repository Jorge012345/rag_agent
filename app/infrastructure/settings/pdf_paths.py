"""PDF paths config (same files as project1, under app/resources/pdfs/)."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PdfPathsSettings(BaseSettings):
    """PDF paths for loading (local dev / demos)."""

    PDF_PATHS: str

    @field_validator("PDF_PATHS", mode="before")
    @classmethod
    def strip_paths(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    def pdf_path_list(self) -> list[str]:
        return [p.strip() for p in self.PDF_PATHS.split(",") if p.strip()]

    model_config = SettingsConfigDict(
        env_file="app/env/pdf_paths.env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
