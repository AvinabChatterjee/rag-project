import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.graph.routing import route_after_execution
from app.graph.workflow import build_workflow_graph
from app.graph.nodes.code_executor import code_executor_node
from app.graph.nodes.init_workflow import init_workflow_node


class RouteAfterExecutionTests(unittest.TestCase):
    def test_success_routes_to_analyze(self) -> None:
        state = {
            "execution_result": {"success": True},
            "execution_attempts": 1,
            "max_execution_attempts": 2,
        }
        self.assertEqual(route_after_execution(state), "analyze")

    def test_first_failure_routes_to_retry(self) -> None:
        state = {
            "execution_result": {"success": False, "error": "KeyError"},
            "execution_attempts": 1,
            "max_execution_attempts": 2,
        }
        self.assertEqual(route_after_execution(state), "retry")

    def test_second_failure_routes_to_analyze(self) -> None:
        state = {
            "execution_result": {"success": False, "error": "KeyError"},
            "execution_attempts": 2,
            "max_execution_attempts": 2,
        }
        self.assertEqual(route_after_execution(state), "analyze")


class RetryFlowSimulationTests(unittest.TestCase):
    def test_executor_failure_then_success_respects_two_attempt_cap(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            csv_path.write_text("region,revenue\nNorth,100\nSouth,200\n", encoding="utf-8")

            state = {
                "execution_attempts": 0,
                "max_execution_attempts": 2,
                "selected_file_path": str(csv_path),
                "planner_output": {"pandas_query": "df['missing'].sum()"},
                "metadata": {"agent_trace": []},
            }

            first = code_executor_node(state)
            state.update(first)
            self.assertFalse(first["execution_result"]["success"])
            self.assertEqual(route_after_execution(state), "retry")

            state["planner_output"]["pandas_query"] = "df['revenue'].sum()"
            second = code_executor_node(state)
            state.update(second)
            self.assertTrue(second["execution_result"]["success"])
            self.assertEqual(second["execution_attempts"], 2)
            self.assertEqual(route_after_execution(state), "analyze")

    def test_init_workflow_sets_max_execution_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sales.csv").write_text("region,revenue\nNorth,100\n", encoding="utf-8")
            state = init_workflow_node(
                {
                    "user_question": "total revenue?",
                    "data_folder": folder,
                }
            )
        self.assertEqual(state["max_execution_attempts"], 2)
        self.assertEqual(state["execution_attempts"], 0)


class WorkflowRetryGraphTests(unittest.TestCase):
    def test_graph_wires_executor_retry_loop(self) -> None:
        graph = build_workflow_graph().compile()
        node_names = set(graph.get_graph().nodes.keys())

        self.assertIn("code_executor", node_names)
        self.assertIn("fix_pandas_query", node_names)
        self.assertIn("data_analyst", node_names)

    @patch("app.graph.workflow.fix_pandas_query_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.generate_pandas_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.query_planner_node", new_callable=AsyncMock)
    def test_graph_visits_fix_node_on_first_execution_failure(
        self,
        mock_query_planner: AsyncMock,
        mock_generate_pandas: AsyncMock,
        mock_fix_pandas_query: AsyncMock,
    ) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            csv_path.write_text("region,revenue\nNorth,100\nSouth,200\n", encoding="utf-8")

            async def planner_side_effect(state):
                return {
                    "status": "planning",
                    "route": "tabular",
                    "selected_file_path": str(csv_path),
                    "selected_file_type": "csv",
                    "planner_output": {
                        "route": "tabular",
                        "reasoning": "test",
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

            async def generate_side_effect(state):
                return {
                    "status": "executing",
                    "planner_output": {
                        **(state.get("planner_output") or {}),
                        "pandas_query": "df['missing'].sum()",
                        "queries": ["df['missing'].sum()"],
                    },
                    "metadata": {
                        **state.get("metadata", {}),
                        "agent_trace": [
                            *state.get("metadata", {}).get("agent_trace", []),
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
                        **state.get("metadata", {}),
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
                        "user_question": "What is total revenue?",
                        "data_folder": folder,
                    }
                )
            )

        visited = [
            entry["node"]
            for entry in final_state["metadata"]["agent_trace"]
            if "node" in entry
        ]
        self.assertIn("code_executor", visited)
        self.assertIn("fix_pandas_query", visited)
        self.assertEqual(visited.count("code_executor"), 2)
        self.assertTrue(final_state["execution_result"]["success"])
        self.assertEqual(final_state["execution_attempts"], 2)


if __name__ == "__main__":
    unittest.main()
