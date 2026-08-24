import unittest
from unittest.mock import MagicMock, patch

from app.graph.nodes.reranker import reranker_node
from app.rag import reranker as reranker_module


def _sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "chunk-1",
            "text": "Plastics must be recycled by 2026.",
            "score": 0.7,
            "file_path": "C:/tmp/policy.pdf",
            "chunk_index": 0,
        },
        {
            "chunk_id": "chunk-2",
            "text": "Revenue grew ten percent last quarter.",
            "score": 0.8,
            "file_path": "C:/tmp/policy.pdf",
            "chunk_index": 1,
        },
        {
            "chunk_id": "chunk-3",
            "text": "All packaging must be recyclable.",
            "score": 0.6,
            "file_path": "C:/tmp/policy.pdf",
            "chunk_index": 2,
        },
    ]


class Phase5RerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        reranker_module.get_cross_encoder.cache_clear()

    def tearDown(self) -> None:
        reranker_module.get_cross_encoder.cache_clear()

    def test_rerank_chunks_returns_top_k_by_cross_encoder_score(self) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.9, 0.95]

        with patch(
            "app.rag.reranker.get_cross_encoder",
            return_value=mock_model,
        ):
            reranked = reranker_module.rerank_chunks(
                "plastics policy requirements",
                _sample_chunks(),
                top_k=2,
            )

        mock_model.predict.assert_called_once()
        self.assertEqual(len(reranked), 2)
        self.assertEqual(reranked[0]["chunk_id"], "chunk-3")
        self.assertEqual(reranked[1]["chunk_id"], "chunk-2")
        self.assertAlmostEqual(reranked[0]["score"], 0.95)
        self.assertAlmostEqual(reranked[1]["score"], 0.9)

    def test_rerank_chunks_returns_empty_for_no_chunks(self) -> None:
        reranked = reranker_module.rerank_chunks(
            "plastics policy requirements",
            [],
            top_k=5,
        )
        self.assertEqual(reranked, [])

    def test_reranker_node_updates_retrieval_result(self) -> None:
        state = {
            "user_question": "What is the plastics policy?",
            "planner_output": {
                "retrieval_query": "plastics policy requirements",
            },
            "retrieval_result": {
                "query_text": "plastics policy requirements",
                "retrieved_chunks": _sample_chunks(),
            },
            "metadata": {"agent_trace": []},
        }

        with patch(
            "app.graph.nodes.reranker.rerank_chunks",
            return_value=_sample_chunks()[:2],
        ) as mock_rerank:
            result = reranker_node(state)

        mock_rerank.assert_called_once_with(
            "plastics policy requirements",
            _sample_chunks(),
        )
        self.assertEqual(len(result["retrieval_result"]["reranked_chunks"]), 2)

    def test_reranker_node_requires_query(self) -> None:
        state = {
            "retrieval_result": {"retrieved_chunks": _sample_chunks()},
            "metadata": {"agent_trace": []},
        }

        with self.assertRaises(ValueError):
            reranker_node(state)


if __name__ == "__main__":
    unittest.main()
