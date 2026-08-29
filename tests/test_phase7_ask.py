import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api.routes import (
    AskRequest,
    ask,
    build_ask_input_state,
    build_ask_response,
)
from app.main import app


def _tabular_final_state(folder: str, csv_path: Path) -> dict:
    return {
        "workflow_id": "wf-tabular-123",
        "status": "completed",
        "user_question": "What is the total revenue?",
        "data_folder": folder,
        "available_files": [
            {
                "file_path": str(csv_path.resolve()),
                "file_name": csv_path.name,
                "file_type": "csv",
            }
        ],
        "route": "tabular",
        "selected_file_path": str(csv_path.resolve()),
        "execution_result": {
            "success": True,
            "attempts": 1,
            "raw_result": 300,
            "executed_code": "df['revenue'].sum()",
            "error": None,
        },
        "analyst_output": {
            "final_answer": "Total revenue is $300.",
            "error_message": None,
            "confidence": "high",
        },
    }


class Phase7AskMappingTests(unittest.TestCase):
    def test_build_ask_input_state_strips_question(self) -> None:
        state = build_ask_input_state("  What is revenue?  ", "/tmp/data")
        self.assertEqual(state["user_question"], "What is revenue?")
        self.assertEqual(state["data_folder"], "/tmp/data")

    def test_build_ask_input_state_rejects_blank_question(self) -> None:
        with self.assertRaises(ValueError):
            build_ask_input_state("   ")

    def test_build_ask_input_state_rejects_blank_data_folder(self) -> None:
        with self.assertRaises(ValueError):
            build_ask_input_state("What is revenue?", "   ")

    def test_build_ask_response_maps_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            final_state = _tabular_final_state(folder, csv_path)

            response = build_ask_response(final_state)

        self.assertEqual(response.workflow_id, "wf-tabular-123")
        self.assertEqual(response.route, "tabular")
        self.assertEqual(response.answer, "Total revenue is $300.")
        self.assertIsNone(response.error)
        self.assertIsNotNone(response.execution_result)
        self.assertTrue(response.execution_result.success)
        self.assertEqual(response.execution_result.raw_result, 300)
        self.assertIsNotNone(response.analyst_output)
        self.assertEqual(response.analyst_output.confidence, "high")


class Phase7AskEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_ask_maps_tabular_workflow_response(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            csv_path.write_text("region,revenue\nNorth,100\nSouth,200\n", encoding="utf-8")
            final_state = _tabular_final_state(folder, csv_path)

            with patch(
                "app.api.routes.get_rag_graph",
            ) as mock_get_graph:
                mock_graph = mock_get_graph.return_value
                mock_graph.ainvoke = AsyncMock(return_value=final_state)
                response = self.client.post(
                    "/ask",
                    json={
                        "question": "What is the total revenue?",
                        "data_folder": folder,
                    },
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["workflow_id"], "wf-tabular-123")
            self.assertEqual(body["route"], "tabular")
            self.assertEqual(body["answer"], "Total revenue is $300.")
            self.assertIsNone(body["error"])
            self.assertTrue(body["execution_result"]["success"])
            self.assertEqual(body["execution_result"]["raw_result"], 300)
            self.assertEqual(body["analyst_output"]["confidence"], "high")
            mock_graph.ainvoke.assert_awaited_once()
            invoke_args = mock_graph.ainvoke.await_args
            self.assertEqual(
                invoke_args.args[0],
                {
                    "user_question": "What is the total revenue?",
                    "data_folder": folder,
                    "workflow_id": invoke_args.args[0]["workflow_id"],
                },
            )
            self.assertEqual(
                invoke_args.args[0]["workflow_id"],
                invoke_args.args[1]["configurable"]["thread_id"],
            )

    def test_ask_rejects_empty_data_folder(self) -> None:
        response = self.client.post(
            "/ask",
            json={"question": "What is revenue?", "data_folder": "   "},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("data_folder", response.json()["detail"])

    def test_ask_rejects_blank_question(self) -> None:
        response = self.client.post("/ask", json={"question": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("question is required", response.json()["detail"])

    def test_ask_returns_400_when_data_folder_has_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            response = self.client.post(
                "/ask",
                json={"question": "What is revenue?", "data_folder": folder},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("No supported files found", response.json()["detail"])

    def test_ask_handler_returns_mapped_response(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            final_state = _tabular_final_state(folder, csv_path)

            with patch("app.api.routes.get_rag_graph") as mock_get_graph:
                mock_graph = mock_get_graph.return_value
                mock_graph.ainvoke = AsyncMock(return_value=final_state)
                response = asyncio.run(
                    ask(
                        AskRequest(
                            question="What is the total revenue?",
                            data_folder=folder,
                        )
                    )
                )

        self.assertEqual(response.workflow_id, "wf-tabular-123")
        self.assertEqual(response.route, "tabular")
        self.assertEqual(response.answer, "Total revenue is $300.")


if __name__ == "__main__":
    unittest.main()
