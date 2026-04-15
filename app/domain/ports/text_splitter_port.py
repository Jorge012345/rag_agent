"""Port: split loaded documents into smaller chunks."""

from abc import ABC, abstractmethod
from typing import List

from app.domain.entities.raw_document import RawDocument


class TextSplitterPort(ABC):
    """Abstraction over text-splitting strategy."""

    @abstractmethod
    def split_documents(self, documents: List[RawDocument]) -> List[RawDocument]:
        """Split input documents into chunked documents."""
