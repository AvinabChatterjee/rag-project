import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import Settings, get_settings
from app.graph.nodes.retriever import retriever_node
from app.rag import retriever as retriever_module
from app.rag.ingest import get_chroma_client


def _make_state(**overrides) -> dict:
    base = {
        "selected_file_path": "C:/tmp/policy.pdf",
        "planner_output": {
            "retrieval_query": "plastics policy requirements",
        },
        "retrieval_result": {
            "query_embedding": [1.0, 0.0, 0.0],
            "query_text": "plastics policy requirements",
        },
        "metadata": {"agent_trace": []},
    }
    base.update(overrides)
    return base


class Phase5RetrieverTests(unittest.TestCase):
    def test_format_chroma_results_maps_chunks(self) -> None:
        file_path = "C:/tmp/policy.pdf"
        results = {
            "ids": [["C:/tmp/policy.pdf::chunk::0"]],
            "documents": [["Plastics must be recycled by 2026."]],
            "metadatas": [[{"file_path": file_path, "chunk_index": 0}]],
            "distances": [[0.25]],
        }

        chunks = retriever_module._format_chroma_results(results, file_path=file_path)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["chunk_id"], "C:/tmp/policy.pdf::chunk::0")
        self.assertEqual(chunks[0]["text"], "Plastics must be recycled by 2026.")
        self.assertEqual(chunks[0]["file_path"], file_path)
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertAlmostEqual(chunks[0]["score"], 0.8)

    def test_retrieve_chunks_queries_chroma_with_file_path_filter(self) -> None:
        file_path = Path("C:/tmp/policy.pdf")
        query_embedding = [1.0, 0.0, 0.0]

        with patch("app.rag.retriever.get_chroma_collection") as mock_get_collection:
            mock_collection = mock_get_collection.return_value
            mock_collection.query.return_value = {
                "ids": [["C:/tmp/policy.pdf::chunk::0"]],
                "documents": [["Relevant policy text."]],
                "metadatas": [[{"file_path": str(file_path), "chunk_index": 0}]],
                "distances": [[0.1]],
            }

            chunks = retriever_module.retrieve_chunks(file_path, query_embedding, n_results=5)

        mock_collection.query.assert_called_once_with(
            query_embeddings=[query_embedding],
            n_results=5,
            where={"file_path": str(file_path.resolve())},
            include=["documents", "metadatas", "distances"],
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Relevant policy text.")

    def test_retriever_node_uses_query_embedding_from_cache_lookup(self) -> None:
        with patch(
            "app.graph.nodes.retriever.retrieve_chunks",
            return_value=[
                {
                    "chunk_id": "chunk-1",
                    "text": "Policy excerpt.",
                    "score": 0.9,
                    "file_path": "C:/tmp/policy.pdf",
                    "chunk_index": 0,
                }
            ],
        ) as mock_retrieve:
            result = asyncio.run(retriever_node(_make_state()))

        mock_retrieve.assert_called_once_with(
            str(Path("C:/tmp/policy.pdf").resolve()),
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(len(result["retrieval_result"]["retrieved_chunks"]), 1)
        self.assertEqual(
            result["retrieval_result"]["retrieved_chunks"][0]["text"],
            "Policy excerpt.",
        )

    def test_retriever_node_embeds_when_query_embedding_missing(self) -> None:
        async def fake_embed(text: str) -> list[float]:
            self.assertEqual(text, "plastics policy requirements")
            return [0.0, 1.0, 0.0]

        with (
            patch(
                "app.graph.nodes.retriever.embed_text",
                new=fake_embed,
            ),
            patch(
                "app.graph.nodes.retriever.retrieve_chunks",
                return_value=[],
            ) as mock_retrieve,
        ):
            result = asyncio.run(
                retriever_node(
                    _make_state(
                        retrieval_result={},
                    )
                )
            )

        mock_retrieve.assert_called_once_with(
            str(Path("C:/tmp/policy.pdf").resolve()),
            [0.0, 1.0, 0.0],
        )
        self.assertEqual(result["retrieval_result"]["query_embedding"], [0.0, 1.0, 0.0])
        self.assertEqual(result["retrieval_result"]["retrieved_chunks"], [])

    def test_ingest_then_retrieve_round_trip(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
            vector_db_dir = Path(folder) / "vector_db"
            path = Path(folder) / "policy.txt"
            path.write_text(
                "Plastics policy requires all packaging to be recyclable by 2026.",
                encoding="utf-8",
            )
            settings = Settings(vector_db_dir=vector_db_dir, retriever_top_k=3)
            get_settings.cache_clear()
            get_chroma_client.cache_clear()

            async def fake_embed(texts: list[str]) -> list[list[float]]:
                return [[float(index), 0.0, 0.0] for index, _ in enumerate(texts)]

            try:
                with patch("app.rag.ingest.get_settings", return_value=settings), patch(
                    "app.rag.retriever.get_settings",
                    return_value=settings,
                ), patch("app.rag.ingest.embed_texts", new=fake_embed):
                    from app.rag.ingest import ingest_document

                    indexed = asyncio.run(ingest_document(path))
                    self.assertGreaterEqual(indexed, 1)

                    chunks = retriever_module.retrieve_chunks(
                        path,
                        [0.0, 0.0, 0.0],
                        n_results=3,
                    )
            finally:
                get_chroma_client.cache_clear()
                get_settings.cache_clear()

            self.assertGreaterEqual(len(chunks), 1)
            self.assertIn("recyclable", chunks[0]["text"].lower())


if __name__ == "__main__":
    unittest.main()
