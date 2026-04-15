"""Plain document loaded from a file (framework-agnostic)."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RawDocument:
    """One logical piece of text + metadata (e.g. one PDF page)."""

    page_content: str
    metadata: Mapping[str, Any]
