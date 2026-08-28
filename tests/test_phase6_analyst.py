import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from app.graph.nodes.data_analyst import data_analyst_node
from app.graph.validation import (
    ensure_friendly_error_answer,
    ensure_source_citations,
    finalize_analyst_output,
    normalize_analyst_output,
)
from app.rag.prompts import build_data_analyst_user_prompt


class Phase6AnalystTests(unittest.TestCase):
    def test_build_data_analyst_user_prompt_tabular_success(self) -> None:
        prompt = build_data_analyst_user_prompt(
            "What is total revenue?",
            scenario="tabular_success",
            raw_result=300,
        )

        self.assertIn("User question:", prompt)
        self.assertIn("Execution result:", prompt)
        self.assertIn("300", prompt)
        self.assertIn("Highlight key numbers", prompt)

    def test_build_data_analyst_user_prompt_tabular_failure_includes_friendly_rules(self) -> None:
        prompt = build_data_analyst_user_prompt(
            "What is total revenue?",
            scenario="tabular_failure",
            execution_error="KeyError: 'Region'",
        )

        self.assertIn("friendly final_answer", prompt)
        self.assertIn("Do not invent data", prompt)

    def test_build_data_analyst_user_prompt_document_miss_includes_sources(self) -> None:
        prompt = build_data_analyst_user_prompt(
            "What is the plastics policy?",
            scenario="document_cache_miss",
            llm_answer="All plastics must be recyclable by 2026.",
            sources=[{"chunk_id": "chunk-1", "file_path": "C:/tmp/policy.pdf"}],
        )

        self.assertIn("Document answer:", prompt)
        self.assertIn("Sources:", prompt)
        self.assertIn("policy.pdf", prompt)
        self.assertIn("Cite every source file path", prompt)

    def test_ensure_source_citations_appends_missing_paths(self) -> None:
        cited = ensure_source_citations(
            "Plastics must be recyclable by 2026.",
            [{"chunk_id": "chunk-1", "file_path": "C:/tmp/policy.pdf"}],
        )

        self.assertIn("C:/tmp/policy.pdf", cited)

    def test_ensure_friendly_error_answer_replaces_technical_traceback(self) -> None:
        friendly = ensure_friendly_error_answer("Traceback: KeyError: 'Region'")

        self.assertNotIn("KeyError", friendly)
        self.assertIn("couldn't complete", friendly.lower())

    def test_finalize_analyst_output_enforces_failure_rules(self) -> None:
        finalized = finalize_analyst_output(
            {
                "final_answer": "Traceback: KeyError: 'Region'",
                "error_message": None,
                "confidence": "medium",
            },
            scenario="tabular_failure",
            preset_error_message="KeyError: 'Region' Available columns: region, revenue.",
        )

        self.assertEqual(finalized["confidence"], "low")
        self.assertIn("Available columns: region, revenue", finalized["error_message"])
        self.assertNotIn("KeyError", finalized["final_answer"])

    def test_normalize_analyst_output_enforces_schema(self) -> None:
        normalized = normalize_analyst_output(
            {
                "final_answer": "North leads with $120,000 in revenue.",
                "error_message": None,
                "confidence": "high",
            }
        )

        self.assertEqual(set(normalized.keys()), {"final_answer", "error_message", "confidence"})
        self.assertEqual(normalized["final_answer"], "North leads with $120,000 in revenue.")
        self.assertIsNone(normalized["error_message"])
        self.assertEqual(normalized["confidence"], "high")

    def test_normalize_analyst_output_rejects_extra_keys(self) -> None:
        with self.assertRaises(ValueError):
            normalize_analyst_output(
                {
                    "final_answer": "Answer",
                    "error_message": None,
                    "confidence": "medium",
                    "route": "tabular",
                }
            )

    def test_normalize_analyst_output_defaults_invalid_confidence(self) -> None:
        normalized = normalize_analyst_output(
            {
                "final_answer": "Answer",
                "error_message": None,
                "confidence": "unknown",
            }
        )

        self.assertEqual(normalized["confidence"], "medium")

    def test_data_analyst_node_sets_completed_status_and_schema(self) -> None:
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
        ):
            result = asyncio.run(data_analyst_node(state))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            set(result["analyst_output"].keys()),
            {"final_answer", "error_message", "confidence"},
        )

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
                    "confidence": "medium",
                }
            ),
        ):
            result = asyncio.run(data_analyst_node(state))

        self.assertIn(
            "Available columns: region, revenue",
            result["analyst_output"]["error_message"],
        )
        self.assertEqual(result["analyst_output"]["confidence"], "low")
        self.assertNotIn("KeyError", result["analyst_output"]["final_answer"])

    def test_data_analyst_node_document_cache_miss_appends_source_citation(self) -> None:
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
                    "final_answer": "Plastics must be recyclable by 2026.",
                    "error_message": None,
                    "confidence": "high",
                }
            ),
        ):
            result = asyncio.run(data_analyst_node(state))

        self.assertIn("C:/tmp/policy.pdf", result["analyst_output"]["final_answer"])

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
