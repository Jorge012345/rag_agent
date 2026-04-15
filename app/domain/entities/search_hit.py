"""One result row from vector similarity search."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Matched chunk text, metadata, and optional distance from the query vector."""

    page_content: str
    metadata: Mapping[str, Any]
    distance: Optional[float] = None
