import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.api.routes import build_invoke_config, get_workflow
from app.graph.nodes.init_workflow import init_workflow_node
from app.main import app


class Phase7InitWorkflowTests(unittest.TestCase):
    def test_init_workflow_preserves_preassigned_workflow_id(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "sales.csv").write_text("region,revenue\nNorth,100\n", encoding="utf-8")

            result = init_workflow_node(
                {
                    "user_question": "What is revenue?",
                    "data_folder": folder,
                    "workflow_id": "preset-workflow-id",
                }
            )

        self.assertEqual(result["workflow_id"], "preset-workflow-id")


class Phase7WorkflowEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_get_workflow_returns_persisted_state(self) -> None:
        workflow_state = {
            "workflow_id": "wf-persisted-123",
            "status": "completed",
            "user_question": "What is revenue?",
            "data_folder": "/tmp/data",
            "route": "tabular",
        }
        snapshot = MagicMock()
        snapshot.values = workflow_state
        snapshot.next = ()

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        with patch("app.api.routes.get_rag_graph", return_value=mock_graph):
            response = self.client.get("/workflow/wf-persisted-123")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["workflow_id"], "wf-persisted-123")
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["state"]["route"], "tabular")
        mock_graph.aget_state.assert_awaited_once_with(
            build_invoke_config("wf-persisted-123")
        )

    def test_get_workflow_returns_404_when_missing(self) -> None:
        snapshot = MagicMock()
        snapshot.values = {}
        snapshot.next = ()

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        with patch("app.api.routes.get_rag_graph", return_value=mock_graph):
            response = self.client.get("/workflow/missing-workflow-id")

        self.assertEqual(response.status_code, 404)
        self.assertIn("Workflow not found", response.json()["detail"])

    def test_get_workflow_rejects_blank_id(self) -> None:
        response = self.client.get("/workflow/%20%20")
        self.assertEqual(response.status_code, 400)

    def test_get_workflow_handler_returns_snapshot(self) -> None:
        workflow_state = {
            "workflow_id": "wf-handler-123",
            "status": "completed",
            "user_question": "test",
            "data_folder": "/tmp",
        }
        snapshot = MagicMock()
        snapshot.values = workflow_state
        snapshot.next = ("data_analyst",)

        mock_graph = MagicMock()
        mock_graph.aget_state = AsyncMock(return_value=snapshot)

        with patch("app.api.routes.get_rag_graph", return_value=mock_graph):
            response = asyncio.run(get_workflow("wf-handler-123"))

        self.assertEqual(response.workflow_id, "wf-handler-123")
        self.assertEqual(response.next, ["data_analyst"])


if __name__ == "__main__":
    unittest.main()
