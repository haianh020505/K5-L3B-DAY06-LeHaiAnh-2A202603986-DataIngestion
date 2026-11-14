"""Persistent vector store on Chroma (cosine, HNSW approximate search) with metadata filters."""
from pathlib import Path
from typing import Any

import chromadb

from .models import Chunk

FILTERABLE = {"chapter", "section"}


class ChromaStore:
    def __init__(self, path: Path, dimensions: int) -> None:
        self.client = chromadb.PersistentClient(path=str(path))
        self.dimensions = dimensions

    def _collection(self, strategy: str):
        return self.client.get_or_create_collection(
            f"day06_{strategy}", metadata={"hnsw:space": "cosine"})

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict[str, int]:
        """Upsert by stable chunk ID and delete this document's chunks that no longer exist."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError("vector dimension does not match the embedder")
        col = self._collection(strategy)
        new_ids = {c.chunk_id for c in chunks}
        if len(new_ids) != len(chunks):
            raise ValueError("duplicate chunk IDs")
        stale = []
        for doc_id in {c.document_id for c in chunks}:
            stale += [i for i in col.get(where={"document_id": doc_id}, include=[])["ids"] if i not in new_ids]
        if stale:
            col.delete(ids=stale)
        step = self.client.get_max_batch_size()
        for i in range(0, len(chunks), step):
            part = chunks[i:i + step]
            col.upsert(
                ids=[c.chunk_id for c in part],
                embeddings=vectors[i:i + step],
                documents=[c.text for c in part],
                metadatas=[{"document_id": c.document_id, "source": c.source, "page": c.page, **c.metadata}
                           for c in part],
            )
        return {"upserted": len(chunks), "deleted_stale": len(stale), "count_after": col.count()}

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Top-k by cosine; filter values are passed as data to Chroma's where clause."""
        if filters and not set(filters) <= FILTERABLE:
            raise ValueError(f"filter keys must be in {FILTERABLE}")
        where = None
        if filters:
            clauses = [{k: v} for k, v in filters.items()]
            where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
        res = self._collection(strategy).query(
            query_embeddings=[query_vector], n_results=top_k, where=where,
            include=["documents", "metadatas", "distances"])
        return [
            {"rank": r + 1, "chunk_id": cid, "score": round(1 - dist, 6), "text": doc, "metadata": meta}
            for r, (cid, doc, meta, dist) in enumerate(zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]))
        ]
