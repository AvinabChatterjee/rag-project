from __future__ import annotations

from pathlib import Path
from typing import Any

from app.config import get_settings
from app.rag.ingest import get_chroma_collection


def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + distance)


def _format_chroma_results(
    results: dict[str, Any],
    *,
    file_path: str,
) -> list[dict[str, Any]]:
    ids = results.get("ids") or [[]]
    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not ids or not ids[0]:
        return []

    chunks: list[dict[str, Any]] = []
    for chunk_id, text, metadata, distance in zip(
        ids[0],
        documents[0],
        metadatas[0],
        distances[0],
        strict=True,
    ):
        chunk_metadata = metadata or {}
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "score": _distance_to_score(float(distance)),
                "file_path": chunk_metadata.get("file_path", file_path),
                "chunk_index": chunk_metadata.get("chunk_index"),
            }
        )
    return chunks


def retrieve_chunks(
    file_path: str | Path,
    query_embedding: list[float],
    *,
    n_results: int | None = None,
) -> list[dict[str, Any]]:
    """Query Chroma for the top-k chunks scoped to the given file_path."""
    if not query_embedding:
        raise ValueError("query_embedding must not be empty.")

    settings = get_settings()
    resolved_path = str(Path(file_path).resolve())
    top_k = n_results if n_results is not None else settings.retriever_top_k

    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"file_path": resolved_path},
        include=["documents", "metadatas", "distances"],
    )
    return _format_chroma_results(results, file_path=resolved_path)
