"""Chunk of text with its embedding vector (domain value object)."""

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Text chunk + metadata + embedding (immutable)."""

    page_content: str
    metadata: Mapping[str, Any]
    embedding: Tuple[float, ...]
