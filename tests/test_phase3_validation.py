import unittest

from app.graph.validation import (
    parse_pandas_response,
    parse_planner_response,
    require_successful_dataset_summary,
    validate_pandas_query,
)


class Phase3ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.files = [
            {
                "file_path": "C:/data/sales.csv",
                "file_name": "sales.csv",
                "file_type": "csv",
            },
            {
                "file_path": "C:/data/policy.pdf",
                "file_name": "policy.pdf",
                "file_type": "document",
            },
        ]
        self.summary = {
            "status": "success",
            "dtypes": {"region": "object", "revenue": "int64"},
        }

    def test_parse_planner_response_success(self) -> None:
        route, selected, reasoning, retrieval_query = parse_planner_response(
            {
                "route": "tabular",
                "reasoning": "needs aggregation",
                "selected_file_path": "sales.csv",
                "retrieval_query": None,
            },
            self.files,
            "total revenue?",
        )
        self.assertEqual(route, "tabular")
        self.assertEqual(selected["file_name"], "sales.csv")
        self.assertIsNone(retrieval_query)
        self.assertEqual(reasoning, "needs aggregation")

    def test_parse_planner_response_rejects_unknown_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown file"):
            parse_planner_response(
                {
                    "route": "tabular",
                    "reasoning": "test",
                    "selected_file_path": "missing.csv",
                },
                self.files,
                "question",
            )

    def test_parse_planner_response_rejects_route_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_planner_response(
                {
                    "route": "document",
                    "reasoning": "test",
                    "selected_file_path": "sales.csv",
                },
                self.files,
                "question",
            )

    def test_validate_pandas_query_rejects_unknown_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown columns"):
            validate_pandas_query("df['Region'].sum()", self.summary)

    def test_validate_pandas_query_accepts_known_columns(self) -> None:
        query = validate_pandas_query("df['revenue'].sum()", self.summary)
        self.assertEqual(query, "df['revenue'].sum()")

    def test_validate_pandas_query_rejects_file_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "read_csv"):
            validate_pandas_query("pd.read_csv('x.csv')", self.summary)

    def test_parse_pandas_response_success(self) -> None:
        query = parse_pandas_response(
            {"pandas_query": "df['region'].nunique()"},
            self.summary,
        )
        self.assertEqual(query, "df['region'].nunique()")

    def test_require_successful_dataset_summary(self) -> None:
        with self.assertRaisesRegex(ValueError, "Dataset inspection failed"):
            require_successful_dataset_summary(
                {"status": "error", "error": "Dataset inspection failed."}
            )


if __name__ == "__main__":
    unittest.main()
