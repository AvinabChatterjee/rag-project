import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.api.routes import AskRequest, ask
from app.graph.workflow import build_workflow_graph, rag_graph
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


def _tabular_planner_state(csv_path: Path, state: dict) -> dict:
    return {
        "status": "planning",
        "route": "tabular",
        "selected_file_path": str(csv_path),
        "selected_file_type": "csv",
        "planner_output": {
            "route": "tabular",
            "reasoning": "test tabular route",
            "dataset_summary": None,
            "pandas_query": None,
            "retrieval_query": None,
            "queries": [],
        },
        "metadata": {
            **state.get("metadata", {}),
            "agent_trace": [
                *state.get("metadata", {}).get("agent_trace", []),
                {"node": "query_planner"},
            ],
        },
    }


def _inspect_state(state: dict, csv_path: Path) -> dict:
    from app.graph.nodes.inspect_dataset import inspect_dataset_node

    inspected = inspect_dataset_node(state)
    return {
        **inspected,
        "metadata": {
            **inspected.get("metadata", {}),
            "agent_trace": [
                *state.get("metadata", {}).get("agent_trace", []),
                {"node": "inspect_dataset"},
            ],
        },
    }


@unittest.skipUnless(_api_key_configured(), "OPENAI_API_KEY not configured")
class Phase4WorkflowTests(unittest.TestCase):
    def test_valid_csv_execution_success(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sales.csv").write_text(
                "region,revenue\nNorth,100\nSouth,200\nEast,50\n",
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

        execution_result = final_state.get("execution_result") or {}
        self.assertEqual(final_state["route"], "tabular")
        self.assertTrue(execution_result.get("success"))
        self.assertIsNotNone(execution_result.get("raw_result"))
        self.assertNotIn(
            "Phase 2 stub execution result",
            str(execution_result.get("raw_result")),
        )
        self.assertIn("code_executor", _visited_nodes(final_state))

    def test_ask_returns_real_execution_result(self) -> None:
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
        self.assertIsNotNone(response.execution_result)
        self.assertTrue(response.execution_result.success)
        self.assertIsNotNone(response.execution_result.raw_result)
        self.assertNotIn("Phase 2 stubs", response.message)

    @patch("app.graph.workflow.fix_pandas_query_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.generate_pandas_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.query_planner_node", new_callable=AsyncMock)
    def test_retry_trace_on_forced_execution_failure(
        self,
        mock_query_planner: AsyncMock,
        mock_generate_pandas: AsyncMock,
        mock_fix_pandas_query: AsyncMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            csv_path.write_text("region,revenue\nNorth,100\nSouth,200\n", encoding="utf-8")

            async def planner_side_effect(state):
                return _tabular_planner_state(csv_path, state)

            async def generate_side_effect(state):
                inspected = _inspect_state(state, csv_path)
                return {
                    "status": "executing",
                    "planner_output": {
                        **(inspected.get("planner_output") or {}),
                        "pandas_query": "df['Region'].sum()",
                        "queries": ["df['Region'].sum()"],
                    },
                    "metadata": {
                        **inspected.get("metadata", {}),
                        "agent_trace": [
                            *inspected.get("metadata", {}).get("agent_trace", []),
                            {"node": "generate_pandas"},
                        ],
                    },
                }

            async def fix_side_effect(state):
                return {
                    "status": "executing",
                    "planner_output": {
                        **(state.get("planner_output") or {}),
                        "pandas_query": "df['revenue'].sum()",
                        "queries": [
                            *(state.get("planner_output", {}).get("queries") or []),
                            "df['revenue'].sum()",
                        ],
                    },
                    "metadata": {
                        **(state.get("metadata", {})),
                        "agent_trace": [
                            *state.get("metadata", {}).get("agent_trace", []),
                            {"node": "fix_pandas_query"},
                        ],
                    },
                }

            mock_query_planner.side_effect = planner_side_effect
            mock_generate_pandas.side_effect = generate_side_effect
            mock_fix_pandas_query.side_effect = fix_side_effect

            graph = build_workflow_graph().compile()
            final_state = asyncio.run(
                graph.ainvoke(
                    {
                        "user_question": "What is the total revenue across all regions?",
                        "data_folder": folder,
                    }
                )
            )

        visited = _visited_nodes(final_state)
        executor_indexes = [index for index, node in enumerate(visited) if node == "code_executor"]
        fix_index = visited.index("fix_pandas_query")

        self.assertEqual(len(executor_indexes), 2)
        self.assertLess(executor_indexes[0], fix_index)
        self.assertLess(fix_index, executor_indexes[1])
        self.assertTrue(final_state["execution_result"]["success"])
        self.assertEqual(final_state["execution_result"]["raw_result"], 300)

    @patch("app.graph.workflow.fix_pandas_query_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.generate_pandas_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.query_planner_node", new_callable=AsyncMock)
    def test_double_execution_failure_passes_error_to_analyst(
        self,
        mock_query_planner: AsyncMock,
        mock_generate_pandas: AsyncMock,
        mock_fix_pandas_query: AsyncMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            csv_path.write_text("region,revenue\nNorth,100\nSouth,200\n", encoding="utf-8")

            async def planner_side_effect(state):
                return _tabular_planner_state(csv_path, state)

            async def generate_side_effect(state):
                inspected = _inspect_state(state, csv_path)
                return {
                    "status": "executing",
                    "planner_output": {
                        **(inspected.get("planner_output") or {}),
                        "pandas_query": "df['Region'].sum()",
                        "queries": ["df['Region'].sum()"],
                    },
                    "metadata": {
                        **inspected.get("metadata", {}),
                        "agent_trace": [
                            *inspected.get("metadata", {}).get("agent_trace", []),
                            {"node": "generate_pandas"},
                        ],
                    },
                }

            async def fix_side_effect(state):
                return {
                    "status": "executing",
                    "planner_output": {
                        **(state.get("planner_output") or {}),
                        "pandas_query": "df['missing'].sum()",
                        "queries": [
                            *(state.get("planner_output", {}).get("queries") or []),
                            "df['missing'].sum()",
                        ],
                    },
                    "metadata": {
                        **(state.get("metadata", {})),
                        "agent_trace": [
                            *state.get("metadata", {}).get("agent_trace", []),
                            {"node": "fix_pandas_query"},
                        ],
                    },
                }

            mock_query_planner.side_effect = planner_side_effect
            mock_generate_pandas.side_effect = generate_side_effect
            mock_fix_pandas_query.side_effect = fix_side_effect

            graph = build_workflow_graph().compile()
            final_state = asyncio.run(
                graph.ainvoke(
                    {
                        "user_question": "What is the total revenue across all regions?",
                        "data_folder": folder,
                    }
                )
            )

        execution_result = final_state.get("execution_result") or {}
        analyst_output = final_state.get("analyst_output") or {}

        self.assertFalse(execution_result.get("success"))
        self.assertEqual(final_state.get("execution_attempts"), 2)
        self.assertEqual(final_state.get("status"), "completed")
        self.assertIn("fix_pandas_query", _visited_nodes(final_state))
        self.assertIn("Available columns: region, revenue", analyst_output.get("error_message", ""))


if __name__ == "__main__":
    unittest.main()
