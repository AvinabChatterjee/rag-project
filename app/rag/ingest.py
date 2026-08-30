from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
import tiktoken
from chromadb.api.models.Collection import Collection

from app.config import get_settings
from app.llm.openai_client import embed_texts
from app.tools.document_tools import load_document_text
from app.utils.file_utils import detect_file_type, validate_local_file

_ENCODING_NAME = "cl100k_base"


@lru_cache
def get_chroma_client() -> chromadb.PersistentClient:
    settings = get_settings()
    return chromadb.PersistentClient(path=str(settings.vector_db_dir))


def get_chroma_collection() -> Collection:
    settings = get_settings()
    client = get_chroma_client()
    return client.get_or_create_collection(name=settings.chroma_collection_name)


def _collection_embedding_dimension(collection: Collection) -> int | None:
    sample = collection.get(include=["embeddings"], limit=1)
    embeddings = sample.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        return None
    first = embeddings[0]
    if first is None:
        return None
    return len(first)


def _recreate_chroma_collection() -> Collection:
    settings = get_settings()
    client = get_chroma_client()
    name = settings.chroma_collection_name
    try:
        client.delete_collection(name)
    except (ValueError, Exception):
        pass
    return client.get_or_create_collection(name=name)


def _collection_for_embeddings(embeddings: list[list[float]]) -> Collection:
    if not embeddings:
        raise ValueError("embeddings must not be empty.")

    expected_dim = len(embeddings[0])
    collection = get_chroma_collection()
    current_dim = _collection_embedding_dimension(collection)
    if current_dim is not None and current_dim != expected_dim:
        collection = _recreate_chroma_collection()
    return collection


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping token windows for embedding."""
    settings = get_settings()
    size = chunk_size if chunk_size is not None else settings.chunk_size_tokens
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap_tokens

    if size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    normalized = text.strip()
    if not normalized:
        return []

    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    tokens = encoding.encode(normalized)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunks.append(encoding.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def _delete_existing_chunks(collection: Collection, file_path: str) -> None:
    existing = collection.get(where={"file_path": file_path}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def count_indexed_chunks(file_path: str | Path) -> int:
    """Return how many Chroma chunks are stored for the resolved file_path."""
    resolved_path = str(Path(file_path).resolve())
    collection = get_chroma_collection()
    existing = collection.get(where={"file_path": resolved_path}, include=[])
    return len(existing.get("ids") or [])


async def ingest_document(file_path: str | Path) -> int:
    """Load, chunk, embed, and store a document in Chroma scoped by file_path."""
    path = validate_local_file(file_path)
    if detect_file_type(path) != "document":
        raise ValueError(
            f"Only document files can be ingested (PDF, DOCX, TXT), "
            f"got '{detect_file_type(path)}'."
        )

    text = load_document_text(path)
    if not text.strip():
        raise ValueError(f"Document has no extractable text: {path}")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError(f"Document produced no chunks after processing: {path}")

    embeddings = await embed_texts(chunks)
    resolved_path = str(path.resolve())
    collection = _collection_for_embeddings(embeddings)
    _delete_existing_chunks(collection, resolved_path)

    ids = [f"{resolved_path}::chunk::{index}" for index in range(len(chunks))]
    metadatas = [
        {"file_path": resolved_path, "chunk_index": index}
        for index in range(len(chunks))
    ]
    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return len(chunks)
