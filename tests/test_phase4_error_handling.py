import tempfile
import unittest
from pathlib import Path

from app.graph.nodes.code_executor import code_executor_node
from app.graph.nodes.data_analyst import data_analyst_node
from app.graph.validation import (
    build_execution_error_message,
    require_failed_execution,
    require_pandas_query,
    require_selected_file_path,
)


class Phase4ErrorHandlingTests(unittest.TestCase):
    def test_require_selected_file_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "selected_file_path"):
            require_selected_file_path(None)
        self.assertEqual(require_selected_file_path("/tmp/a.csv"), "/tmp/a.csv")

    def test_require_pandas_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "pandas_query"):
            require_pandas_query({})
        self.assertEqual(
            require_pandas_query({"pandas_query": "df.head()"}),
            "df.head()",
        )

    def test_require_failed_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "failed execution_result"):
            require_failed_execution({"success": True})
        with self.assertRaisesRegex(ValueError, "execution_result.error"):
            require_failed_execution({"success": False})
        self.assertEqual(
            require_failed_execution({"success": False, "error": "KeyError: 'x'"}),
            "KeyError: 'x'",
        )

    def test_build_execution_error_message_includes_columns(self) -> None:
        message = build_execution_error_message(
            "KeyError: 'Region'",
            {"dtypes": {"region": "object", "revenue": "int64"}},
        )
        self.assertIn("Available columns: region, revenue", message)

    def test_code_executor_sets_failed_status_after_final_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            csv_path.write_text("region,revenue\nNorth,100\n", encoding="utf-8")

            state = {
                "execution_attempts": 1,
                "max_execution_attempts": 2,
                "selected_file_path": str(csv_path),
                "planner_output": {"pandas_query": "df['missing'].sum()"},
                "metadata": {"agent_trace": []},
            }
            result = code_executor_node(state)

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["execution_result"]["success"])
        self.assertIsNotNone(result["execution_result"]["error"])

    def test_data_analyst_forwards_execution_error_with_columns(self) -> None:
        result = data_analyst_node(
            {
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
        )

        self.assertIn("Available columns: region, revenue", result["analyst_output"]["error_message"])


if __name__ == "__main__":
    unittest.main()
