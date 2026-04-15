"""RAG: retrieve chunks, then answer with an LLM."""

from dataclasses import dataclass
from typing import List, Optional

from app.application.use_cases.query_documents import QueryDocumentsUseCase
from app.domain.entities.search_hit import SearchHit
from app.domain.ports.llm_port import LlmPort

_MAX_CONTEXT_CHARS = 14_000

_SYSTEM_PROMPT = """You are a helpful assistant answering questions about documents the user has ingested.
Use ONLY the provided context excerpts to support your answer. If the context does not contain enough information, say so clearly instead of guessing.
Write in clear, natural prose. Match the language of the user's question when possible."""


@dataclass(frozen=True, slots=True)
class AskDocumentsResult:
    """Generated answer plus the chunks used as context."""

    answer: str
    context_hits: tuple[SearchHit, ...]


class AskDocumentsUseCase:
    """Runs semantic search, builds a context block, and asks the LLM to answer."""

    def __init__(
        self,
        query_documents: QueryDocumentsUseCase,
        llm: LlmPort,
        default_context_k: int,
    ) -> None:
        self._query_documents = query_documents
        self._llm = llm
        self._default_context_k = default_context_k

    def execute(self, query: str, k: Optional[int] = None) -> AskDocumentsResult:
        text = query.strip()
        if not text:
            raise ValueError("Query text must not be empty.")
        top_k = k if k is not None else self._default_context_k
        if top_k < 1:
            raise ValueError("k must be at least 1.")

        hits = self._query_documents.execute(text, top_k)
        if not hits:
            return AskDocumentsResult(
                answer=(
                    "No indexed passages were found. Load PDFs with POST /api/v1/documents/load "
                    "before asking questions."
                ),
                context_hits=(),
            )

        context_block = _build_context_block(hits)
        user_prompt = (
            f"Context excerpts:\n---\n{context_block}\n---\n\n"
            f"Question: {text}\n\n"
            "Answer based on the context above:"
        )
        answer = self._llm.generate(system=_SYSTEM_PROMPT, user=user_prompt)
        return AskDocumentsResult(answer=answer.strip(), context_hits=tuple(hits))


def _build_context_block(hits: List[SearchHit]) -> str:
    parts: list[str] = []
    total = 0
    for i, h in enumerate(hits, start=1):
        meta = h.metadata
        src = meta.get("source", "unknown")
        page = meta.get("page_label", meta.get("page", ""))
        header = f"[{i}] source={src} page={page}"
        body = h.page_content.strip()
        chunk = f"{header}\n{body}"
        if total + len(chunk) > _MAX_CONTEXT_CHARS:
            remaining = _MAX_CONTEXT_CHARS - total
            if remaining > 200:
                parts.append(chunk[:remaining].rstrip() + "…")
            break
        parts.append(chunk)
        total += len(chunk) + 2
    return "\n\n".join(parts)
