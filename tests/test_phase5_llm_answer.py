import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.graph.nodes.llm_answer import llm_answer_node
from app.rag.prompts import build_document_answer_user_prompt


def _sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "chunk-1",
            "text": "All plastics packaging must be recyclable by 2026.",
            "score": 0.95,
            "file_path": "C:/tmp/policy.pdf",
            "chunk_index": 0,
        },
        {
            "chunk_id": "chunk-2",
            "text": "Companies must submit annual compliance reports.",
            "score": 0.8,
            "file_path": "C:/tmp/policy.pdf",
            "chunk_index": 1,
        },
    ]


class Phase5LlmAnswerTests(unittest.TestCase):
    def test_build_document_answer_user_prompt_formats_context(self) -> None:
        prompt = build_document_answer_user_prompt(
            "When must plastics be recyclable?",
            _sample_chunks(),
        )

        self.assertIn("Context:", prompt)
        self.assertIn("recyclable by 2026", prompt)
        self.assertIn("User Query: When must plastics be recyclable?", prompt)

    def test_llm_answer_node_calls_llm_with_reranked_chunks(self) -> None:
        state = {
            "user_question": "When must plastics be recyclable?",
            "selected_file_path": "C:/tmp/policy.pdf",
            "retrieval_result": {"reranked_chunks": _sample_chunks()},
            "metadata": {"agent_trace": []},
        }

        with patch(
            "app.graph.nodes.llm_answer.call_llm",
            new=AsyncMock(return_value="Plastics must be recyclable by 2026."),
        ) as mock_call_llm:
            result = asyncio.run(llm_answer_node(state))

        mock_call_llm.assert_called_once()
        system_prompt, user_prompt = (
            mock_call_llm.call_args.args[0],
            mock_call_llm.call_args.args[1],
        )
        self.assertIn("Answer only from the provided context", system_prompt)
        self.assertIn("recyclable by 2026", user_prompt)
        self.assertEqual(
            result["retrieval_result"]["llm_answer"],
            "Plastics must be recyclable by 2026.",
        )
        self.assertEqual(len(result["retrieval_result"]["sources"]), 2)
        self.assertEqual(result["retrieval_result"]["sources"][0]["chunk_id"], "chunk-1")

    def test_llm_answer_node_returns_i_dont_know_without_chunks(self) -> None:
        state = {
            "user_question": "When must plastics be recyclable?",
            "selected_file_path": "C:/tmp/policy.pdf",
            "retrieval_result": {"reranked_chunks": []},
            "metadata": {"agent_trace": []},
        }

        with patch(
            "app.graph.nodes.llm_answer.call_llm",
            new=AsyncMock(),
        ) as mock_call_llm:
            result = asyncio.run(llm_answer_node(state))

        mock_call_llm.assert_not_called()
        self.assertEqual(result["retrieval_result"]["llm_answer"], "I don't know.")
        self.assertEqual(result["retrieval_result"]["sources"], [])

    def test_llm_answer_node_requires_user_question(self) -> None:
        state = {
            "retrieval_result": {"reranked_chunks": _sample_chunks()},
            "metadata": {"agent_trace": []},
        }

        with self.assertRaises(ValueError):
            asyncio.run(llm_answer_node(state))


if __name__ == "__main__":
    unittest.main()
