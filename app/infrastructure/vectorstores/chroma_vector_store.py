"""ChromaDB persistence with precomputed embeddings."""

import os
import uuid
from typing import Any, Mapping, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.domain.entities.embedded_chunk import EmbeddedChunk
from app.domain.ports.vector_store_port import VectorStorePort
from app.infrastructure.settings import SETTINGS


def _sanitize_metadata(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str, int, float, or bool."""
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
        elif value is None:
            out[str(key)] = ""
        else:
            out[str(key)] = str(value)
    return out


class ChromaVectorStoreAdapter(VectorStorePort):
    """Stores embedded chunks in a persisted Chroma collection."""

    _BATCH = 500

    def __init__(self) -> None:
        persist = SETTINGS.DATABASE.CHROMA_PERSIST_DIRECTORY
        if not os.path.isabs(persist):
            persist = os.path.abspath(persist)
        self._persist_dir = persist
        self._collection_name = SETTINGS.DATABASE.CHROMA_COLLECTION_NAME

    def replace_all(self, chunks: Sequence[EmbeddedChunk]) -> None:
        os.makedirs(self._persist_dir, exist_ok=True, mode=0o755)
        client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            client.delete_collection(self._collection_name)
        except Exception:
            pass
        collection = client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        chunk_list = list(chunks)
        if not chunk_list:
            return

        for start in range(0, len(chunk_list), self._BATCH):
            batch = chunk_list[start : start + self._BATCH]
            ids = [str(uuid.uuid4()) for _ in batch]
            embeddings = [list(c.embedding) for c in batch]
            documents = [c.page_content for c in batch]
            metadatas = [_sanitize_metadata(c.metadata) for c in batch]
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
