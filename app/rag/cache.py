from __future__ import annotations

import math
import sqlite3
import struct
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from app.config import get_settings

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS cache (
    id              TEXT PRIMARY KEY,
    file_path       TEXT NOT NULL,
    query_text      TEXT NOT NULL,
    query_embedding BLOB NOT NULL,
    answer          TEXT NOT NULL,
    created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_FILE_PATH_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_cache_file_path ON cache(file_path);
"""


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimension.")

    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def _embedding_to_blob(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _blob_to_embedding(blob: bytes) -> list[float]:
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(_CREATE_TABLE_SQL)
    connection.execute(_CREATE_FILE_PATH_INDEX_SQL)


@contextmanager
def _cache_connection():
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(settings.cache_db_path))
    connection.row_factory = sqlite3.Row
    _ensure_schema(connection)
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def purge_expired_cache_entries(ttl_hours: int | None = None) -> int:
    settings = get_settings()
    hours = ttl_hours if ttl_hours is not None else settings.semantic_cache_ttl_hours
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with _cache_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM cache WHERE created_at < ?",
            (cutoff.isoformat(),),
        )
        return cursor.rowcount


def insert_cache_entry(
    file_path: str | Path,
    query_text: str,
    query_embedding: list[float],
    answer: str,
) -> str:
    resolved_path = str(Path(file_path).resolve())
    normalized_query = query_text.strip()
    if not normalized_query:
        raise ValueError("query_text must not be empty.")
    if not query_embedding:
        raise ValueError("query_embedding must not be empty.")
    if not answer.strip():
        raise ValueError("answer must not be empty.")

    entry_id = str(uuid4())
    with _cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO cache (id, file_path, query_text, query_embedding, answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                resolved_path,
                normalized_query,
                _embedding_to_blob(query_embedding),
                answer.strip(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return entry_id


def lookup_semantic_cache(
    file_path: str | Path,
    query_embedding: list[float],
    *,
    threshold: float | None = None,
) -> tuple[bool, str | None, float | None]:
    """
    Return (cache_hit, answer, best_score) for the given file_path embedding.
    """
    if not query_embedding:
        raise ValueError("query_embedding must not be empty.")

    settings = get_settings()
    resolved_path = str(Path(file_path).resolve())
    similarity_threshold = (
        threshold if threshold is not None else settings.semantic_cache_threshold
    )

    purge_expired_cache_entries()

    with _cache_connection() as connection:
        rows = connection.execute(
            """
            SELECT answer, query_embedding
            FROM cache
            WHERE file_path = ?
            """,
            (resolved_path,),
        ).fetchall()

    best_score: float | None = None
    best_answer: str | None = None
    for row in rows:
        cached_embedding = _blob_to_embedding(row["query_embedding"])
        score = cosine_similarity(query_embedding, cached_embedding)
        if best_score is None or score > best_score:
            best_score = score
            best_answer = row["answer"]

    if best_score is not None and best_score >= similarity_threshold:
        return True, best_answer, best_score

    return False, None, best_score
