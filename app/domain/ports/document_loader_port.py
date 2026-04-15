"""Port: load PDFs into domain documents."""

from abc import ABC, abstractmethod
from typing import List

from app.domain.entities.raw_document import RawDocument


class DocumentLoaderPort(ABC):
    """Abstraction over PDF loading (PyPDFLoader lives in infrastructure)."""

    @abstractmethod
    def load_pdf(self, file_path: str) -> List[RawDocument]:
        """Load a single PDF file; may return one row per page."""
