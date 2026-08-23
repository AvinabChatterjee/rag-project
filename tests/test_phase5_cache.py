import asyncio
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.config import Settings, get_settings
from app.graph.nodes.cache_lookup import cache_lookup_node
from app.rag import cache as cache_module


def _make_state(**overrides) -> dict:
    base = {
        "user_question": "What is the plastics policy?",
        "selected_file_path": "C:/tmp/policy.pdf",
        "planner_output": {
            "route": "document",
            "retrieval_query": "plastics policy requirements",
        },
        "metadata": {"agent_trace": []},
    }
    base.update(overrides)
    return base


class Phase5CacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cache_db_path = Path(self.temp_dir.name) / "cache.db"
        self.settings = Settings(
            cache_db_path=self.cache_db_path,
            semantic_cache_threshold=0.92,
            semantic_cache_ttl_hours=24,
        )
        get_settings.cache_clear()

        self.settings_patch = patch(
            "app.rag.cache.get_settings",
            return_value=self.settings,
        )
        self.settings_patch.start()

    def tearDown(self) -> None:
        get_settings.cache_clear()
        self.settings_patch.stop()
        self.temp_dir.cleanup()

    def test_cosine_similarity_identical_vectors(self) -> None:
        vector = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(
            cache_module.cosine_similarity(vector, vector),
            1.0,
        )

    def test_lookup_returns_hit_for_similar_embedding(self) -> None:
        file_path = Path(self.temp_dir.name) / "policy.pdf"
        file_path.write_bytes(b"%PDF-1.4\n")
        embedding = [1.0, 0.0, 0.0]

        cache_module.insert_cache_entry(
            file_path,
            "plastics policy requirements",
            embedding,
            "All plastics must be recycled.",
        )

        hit, answer, score = cache_module.lookup_semantic_cache(
            file_path,
            embedding,
        )

        self.assertTrue(hit)
        self.assertEqual(answer, "All plastics must be recycled.")
        self.assertAlmostEqual(score or 0.0, 1.0)

    def test_lookup_returns_miss_for_different_file_path(self) -> None:
        file_path = Path(self.temp_dir.name) / "policy.pdf"
        other_path = Path(self.temp_dir.name) / "other.pdf"
        embedding = [1.0, 0.0, 0.0]

        cache_module.insert_cache_entry(
            file_path,
            "plastics policy requirements",
            embedding,
            "All plastics must be recycled.",
        )

        hit, answer, score = cache_module.lookup_semantic_cache(
            other_path,
            embedding,
        )

        self.assertFalse(hit)
        self.assertIsNone(answer)
        self.assertIsNone(score)

    def test_lookup_returns_miss_below_threshold(self) -> None:
        file_path = Path(self.temp_dir.name) / "policy.pdf"
        file_path.write_bytes(b"%PDF-1.4\n")

        cache_module.insert_cache_entry(
            file_path,
            "plastics policy requirements",
            [1.0, 0.0, 0.0],
            "All plastics must be recycled.",
        )

        hit, answer, score = cache_module.lookup_semantic_cache(
            file_path,
            [0.0, 1.0, 0.0],
        )

        self.assertFalse(hit)
        self.assertIsNone(answer)
        self.assertLess(score or 0.0, 0.92)

    def test_purge_expired_cache_entries(self) -> None:
        file_path = Path(self.temp_dir.name) / "policy.pdf"
        connection = sqlite3.connect(self.cache_db_path)
        connection.execute(cache_module._CREATE_TABLE_SQL)
        connection.execute(
            """
            INSERT INTO cache (id, file_path, query_text, query_embedding, answer, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "old-entry",
                str(file_path.resolve()),
                "old query",
                cache_module._embedding_to_blob([1.0, 0.0, 0.0]),
                "old answer",
                "2000-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()
        connection.close()

        deleted = cache_module.purge_expired_cache_entries()
        self.assertEqual(deleted, 1)

    def test_cache_lookup_node_hit(self) -> None:
        file_path = Path(self.temp_dir.name) / "policy.pdf"
        file_path.write_bytes(b"%PDF-1.4\n")
        embedding = [1.0, 0.0, 0.0]

        cache_module.insert_cache_entry(
            file_path,
            "plastics policy requirements",
            embedding,
            "Cached plastics policy summary.",
        )

        with patch(
            "app.graph.nodes.cache_lookup.embed_text",
            new=AsyncMock(return_value=embedding),
        ):
            result = asyncio.run(
                cache_lookup_node(
                    _make_state(selected_file_path=str(file_path.resolve()))
                )
            )

        self.assertTrue(result["cache_hit"])
        self.assertEqual(
            result["retrieval_result"]["cached_answer"],
            "Cached plastics policy summary.",
        )
        self.assertEqual(result["retrieval_result"]["query_embedding"], embedding)

    def test_cache_lookup_node_miss_forwards_embedding(self) -> None:
        file_path = Path(self.temp_dir.name) / "policy.pdf"
        file_path.write_bytes(b"%PDF-1.4\n")
        embedding = [0.0, 1.0, 0.0]

        with patch(
            "app.graph.nodes.cache_lookup.embed_text",
            new=AsyncMock(return_value=embedding),
        ):
            result = asyncio.run(
                cache_lookup_node(
                    _make_state(selected_file_path=str(file_path.resolve()))
                )
            )

        self.assertFalse(result["cache_hit"])
        self.assertIsNone(result["retrieval_result"]["cached_answer"])
        self.assertEqual(result["retrieval_result"]["query_embedding"], embedding)
        self.assertEqual(
            result["retrieval_result"]["query_text"],
            "plastics policy requirements",
        )


if __name__ == "__main__":
    unittest.main()
