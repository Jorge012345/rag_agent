"""Load PDFs from configured paths (settings / injection)."""

from typing import List, Optional, Sequence

from app.domain.entities.raw_document import RawDocument
from app.domain.ports.document_loader_port import DocumentLoaderPort
from app.domain.ports.text_splitter_port import TextSplitterPort


class LoadPdfsUseCase:
    """Orchestrates loading PDFs via the document loader port."""

    def __init__(
        self,
        document_loader: DocumentLoaderPort,
        text_splitter: TextSplitterPort,
        default_pdf_paths: Sequence[str],
    ) -> None:
        self._loader = document_loader
        self._splitter = text_splitter
        self._default_pdf_paths = list(default_pdf_paths)

    def execute(self, pdf_paths: Optional[Sequence[str]] = None) -> List[RawDocument]:
        paths = list(pdf_paths) if pdf_paths is not None else self._default_pdf_paths
        documents: List[RawDocument] = []
        for path in paths:
            documents.extend(self._loader.load_pdf(path))
        return self._splitter.split_documents(documents)
