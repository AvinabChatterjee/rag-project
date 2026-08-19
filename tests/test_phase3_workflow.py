import asyncio
import tempfile
import unittest
from pathlib import Path

from app.api.routes import AskRequest, ask
from app.graph.workflow import rag_graph
from app.llm.openai_client import require_api_key


def _api_key_configured() -> bool:
    try:
        require_api_key()
        return True
    except ValueError:
        return False


def _visited_nodes(final_state: dict) -> list[str]:
    trace = final_state.get("metadata", {}).get("agent_trace", [])
    return [entry["node"] for entry in trace]


@unittest.skipUnless(_api_key_configured(), "OPENAI_API_KEY not configured")
class Phase3WorkflowTests(unittest.TestCase):
    def test_tabular_question_generates_real_pandas_query(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sales.csv").write_text(
                "region,revenue\nNorth,100\nSouth,200\n",
                encoding="utf-8",
            )

            final_state = asyncio.run(
                rag_graph.ainvoke(
                    {
                        "user_question": "What is the total revenue across all regions?",
                        "data_folder": folder,
                    }
                )
            )

        planner_output = final_state.get("planner_output") or {}
        dataset_summary = planner_output.get("dataset_summary") or {}
        pandas_query = planner_output.get("pandas_query")

        self.assertEqual(final_state["route"], "tabular")
        self.assertEqual(dataset_summary.get("status"), "success")
        self.assertIn("region", dataset_summary.get("dtypes", {}))
        self.assertIn("revenue", dataset_summary.get("dtypes", {}))
        self.assertIsNotNone(pandas_query)
        self.assertNotIn("Phase 2 stub", pandas_query)
        self.assertNotIn("read_csv", pandas_query.lower())

        visited = _visited_nodes(final_state)
        self.assertEqual(
            visited[:5],
            [
                "init_workflow",
                "query_planner",
                "inspect_dataset",
                "generate_pandas",
                "code_executor",
            ],
        )

    def test_document_question_sets_retrieval_query(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "policy.pdf").write_bytes(b"%PDF-1.4\n")

            final_state = asyncio.run(
                rag_graph.ainvoke(
                    {
                        "user_question": "Summarize the plastics policy requirements.",
                        "data_folder": folder,
                    }
                )
            )

        planner_output = final_state.get("planner_output") or {}

        self.assertEqual(final_state["route"], "document")
        self.assertIsNotNone(planner_output.get("retrieval_query"))
        self.assertTrue(planner_output["retrieval_query"].strip())

        visited = _visited_nodes(final_state)
        self.assertEqual(
            visited,
            [
                "init_workflow",
                "query_planner",
                "cache_lookup",
                "retriever",
                "reranker",
                "llm_answer",
                "cache_store",
                "data_analyst",
            ],
        )

    def test_ask_endpoint_returns_agent1_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sales.csv").write_text(
                "region,revenue\nNorth,100\nSouth,200\n",
                encoding="utf-8",
            )

            response = asyncio.run(
                ask(
                    AskRequest(
                        question="What is the total revenue across all regions?",
                        data_folder=folder,
                    )
                )
            )

        self.assertEqual(response.status, "completed")
        self.assertEqual(response.route, "tabular")
        self.assertIsNotNone(response.selected_file_path)
        self.assertTrue(response.selected_file_path.endswith("sales.csv"))


if __name__ == "__main__":
    unittest.main()
