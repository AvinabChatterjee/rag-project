import asyncio
import tempfile
import unittest
from pathlib import Path

from app.graph.nodes.fix_pandas_query import fix_pandas_query_node
from app.llm.openai_client import require_api_key


def _api_key_configured() -> bool:
    try:
        require_api_key()
        return True
    except ValueError:
        return False


class FixPandasQueryNodeTests(unittest.TestCase):
    def test_requires_execution_error(self) -> None:
        state = {
            "user_question": "What is total revenue?",
            "planner_output": {
                "pandas_query": "df['Region'].sum()",
                "dataset_summary": {
                    "status": "success",
                    "dtypes": {"region": "object", "revenue": "int64"},
                },
            },
            "execution_result": {"success": False},
            "metadata": {"agent_trace": []},
        }

        with self.assertRaisesRegex(ValueError, "execution_result.error"):
            asyncio.run(fix_pandas_query_node(state))

    @unittest.skipUnless(_api_key_configured(), "OPENAI_API_KEY not configured")
    def test_fixes_bad_column_with_llm(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            csv_path.write_text(
                "region,revenue\nNorth,100\nSouth,200\n",
                encoding="utf-8",
            )

            state = {
                "user_question": "What is the total revenue across all regions?",
                "planner_output": {
                    "pandas_query": "df['Region'].sum()",
                    "dataset_summary": {
                        "status": "success",
                        "dtypes": {"region": "object", "revenue": "int64"},
                        "unique_values": {
                            "region": ["North", "South"],
                            "revenue": [100, 200],
                        },
                    },
                    "queries": ["df['Region'].sum()"],
                },
                "execution_result": {
                    "success": False,
                    "error": "KeyError: 'Region'",
                },
                "metadata": {"agent_trace": []},
            }

            result = asyncio.run(fix_pandas_query_node(state))
            fixed_query = result["planner_output"]["pandas_query"]

            self.assertNotIn("'Region'", fixed_query)
            self.assertNotEqual(fixed_query, "df['Region'].sum()")
            self.assertTrue(fixed_query.strip())


if __name__ == "__main__":
    unittest.main()
