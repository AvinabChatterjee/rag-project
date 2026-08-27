import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.graph.nodes.data_analyst import data_analyst_node
from app.rag.prompts import build_data_analyst_user_prompt


class Phase6AnalystTests(unittest.TestCase):
    def test_build_data_analyst_user_prompt_tabular_success(self) -> None:
        prompt = build_data_analyst_user_prompt(
            "What is total revenue?",
            raw_result=300,
        )

        self.assertIn("User question:", prompt)
        self.assertIn("Execution result:", prompt)
        self.assertIn("300", prompt)

    def test_build_data_analyst_user_prompt_document_miss_includes_sources(self) -> None:
        prompt = build_data_analyst_user_prompt(
            "What is the plastics policy?",
            llm_answer="All plastics must be recyclable by 2026.",
            sources=[{"chunk_id": "chunk-1", "file_path": "C:/tmp/policy.pdf"}],
        )

        self.assertIn("Document answer:", prompt)
        self.assertIn("Sources:", prompt)
        self.assertIn("policy.pdf", prompt)

    def test_data_analyst_node_tabular_success(self) -> None:
        state = {
            "user_question": "What is total revenue?",
            "route": "tabular",
            "execution_result": {"success": True, "raw_result": 300},
            "metadata": {"agent_trace": []},
        }

        with patch(
            "app.graph.nodes.data_analyst.call_llm_json",
            new=AsyncMock(
                return_value={
                    "final_answer": "Total revenue is 300.",
                    "error_message": None,
                    "confidence": "high",
                }
            ),
        ) as mock_call_llm:
            result = asyncio.run(data_analyst_node(state))

        mock_call_llm.assert_called_once()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["analyst_output"]["final_answer"], "Total revenue is 300.")
        self.assertIsNone(result["analyst_output"]["error_message"])
        self.assertEqual(result["analyst_output"]["confidence"], "high")

    def test_data_analyst_node_tabular_failure_preserves_column_error(self) -> None:
        state = {
            "user_question": "What is total revenue?",
            "route": "tabular",
            "execution_result": {
                "success": False,
                "error": "KeyError: 'Region'",
            },
            "planner_output": {
                "dataset_summary": {
                    "status": "success",
                    "dtypes": {"region": "object", "revenue": "int64"},
                }
            },
            "metadata": {"agent_trace": []},
        }

        with patch(
            "app.graph.nodes.data_analyst.call_llm_json",
            new=AsyncMock(
                return_value={
                    "final_answer": "I could not run that query on the dataset.",
                    "error_message": "ignored by node",
                    "confidence": "low",
                }
            ),
        ):
            result = asyncio.run(data_analyst_node(state))

        self.assertIn(
            "Available columns: region, revenue",
            result["analyst_output"]["error_message"],
        )

    def test_data_analyst_node_document_cache_hit(self) -> None:
        state = {
            "user_question": "What is the plastics policy?",
            "route": "document",
            "cache_hit": True,
            "retrieval_result": {
                "cached_answer": "All plastics must be recyclable by 2026.",
            },
            "metadata": {"agent_trace": []},
        }

        with patch(
            "app.graph.nodes.data_analyst.call_llm_json",
            new=AsyncMock(
                return_value={
                    "final_answer": "The policy requires recyclable plastics by 2026.",
                    "error_message": None,
                    "confidence": "medium",
                }
            ),
        ) as mock_call_llm:
            result = asyncio.run(data_analyst_node(state))

        user_prompt = mock_call_llm.call_args.args[1]
        self.assertIn("Cached document answer:", user_prompt)
        self.assertEqual(result["analyst_output"]["confidence"], "medium")

    def test_data_analyst_node_document_cache_miss(self) -> None:
        state = {
            "user_question": "What is the plastics policy?",
            "route": "document",
            "cache_hit": False,
            "retrieval_result": {
                "llm_answer": "All plastics must be recyclable by 2026.",
                "sources": [{"chunk_id": "chunk-1", "file_path": "C:/tmp/policy.pdf"}],
            },
            "metadata": {"agent_trace": []},
        }

        with patch(
            "app.graph.nodes.data_analyst.call_llm_json",
            new=AsyncMock(
                return_value={
                    "final_answer": "Plastics must be recyclable by 2026 (C:/tmp/policy.pdf).",
                    "error_message": None,
                    "confidence": "high",
                }
            ),
        ) as mock_call_llm:
            result = asyncio.run(data_analyst_node(state))

        user_prompt = mock_call_llm.call_args.args[1]
        self.assertIn("Document answer:", user_prompt)
        self.assertIn("Sources:", user_prompt)
        self.assertEqual(result["analyst_output"]["confidence"], "high")


if __name__ == "__main__":
    unittest.main()
