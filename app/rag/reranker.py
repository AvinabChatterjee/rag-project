from __future__ import annotations

from functools import lru_cache
from typing import Any

from sentence_transformers import CrossEncoder

from app.config import get_settings


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    settings = get_settings()
    return CrossEncoder(settings.reranker_model)


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank retrieved chunks with a local cross-encoder and keep the top-k."""
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty.")
    if not chunks:
        return []

    settings = get_settings()
    keep = top_k if top_k is not None else settings.reranker_top_k
    pairs = [(normalized_query, str(chunk.get("text") or "")) for chunk in chunks]

    model = get_cross_encoder()
    scores = model.predict(pairs)

    reranked_chunks: list[dict[str, Any]] = []
    for chunk, score in zip(chunks, scores, strict=True):
        updated = dict(chunk)
        updated["score"] = float(score)
        reranked_chunks.append(updated)

    reranked_chunks.sort(key=lambda item: item["score"], reverse=True)
    return reranked_chunks[:keep]
