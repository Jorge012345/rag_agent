"""ChromaDB similarity search (read path)."""

import os
from typing import List, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.domain.entities.search_hit import SearchHit
from app.domain.ports.vector_search_port import VectorSearchPort
from app.infrastructure.settings import SETTINGS


class ChromaVectorSearchAdapter(VectorSearchPort):
    """Queries the same persisted collection as `ChromaVectorStoreAdapter`."""

    def __init__(self) -> None:
        persist = SETTINGS.DATABASE.CHROMA_PERSIST_DIRECTORY
        if not os.path.isabs(persist):
            persist = os.path.abspath(persist)
        self._persist_dir = persist
        self._collection_name = SETTINGS.DATABASE.CHROMA_COLLECTION_NAME

    def search_by_vector(
        self,
        query_embedding: Sequence[float],
        k: int,
    ) -> List[SearchHit]:
        if not os.path.isdir(self._persist_dir):
            raise FileNotFoundError(
                "Chroma persist directory does not exist. Ingest documents first via POST /documents/load."
            )
        client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            collection = client.get_collection(name=self._collection_name)
        except Exception as e:
            raise LookupError(
                "Chroma collection not found. Ingest documents first via POST /documents/load."
            ) from e

        count = collection.count()
        if count == 0:
            return []

        n = min(k, max(count, 1))
        raw = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        docs = (raw.get("documents") or [[]])[0] or []
        metas = (raw.get("metadatas") or [[]])[0] or []
        dists = (raw.get("distances") or [[]])[0] or []

        hits: list[SearchHit] = []
        for i, content in enumerate(docs):
            meta = dict(metas[i]) if i < len(metas) and metas[i] else {}
            dist = dists[i] if i < len(dists) else None
            hits.append(
                SearchHit(
                    page_content=content or "",
                    metadata=meta,
                    distance=float(dist) if dist is not None else None,
                )
            )
        return hits
