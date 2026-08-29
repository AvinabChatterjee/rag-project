"""
Phase 7.6 — Milestone validation (M1–M4) via API smoke and optional live E2E tests.
"""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api.routes import AskRequest, ask
from app.config import Settings, get_settings
from app.graph.workflow import init_workflow_engine, shutdown_workflow_engine
from app.llm.openai_client import require_api_key
from app.main import app


def _compile_checkpointed_graph():
    async def _build():
        await shutdown_workflow_engine()
        await init_workflow_engine()
        from app.graph.workflow import get_rag_graph

        return get_rag_graph()

    return asyncio.run(_build())


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
            "reasoning": "milestone test tabular route",
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


def _tabular_success_state(folder: str, csv_path: Path) -> dict:
    resolved = str(csv_path.resolve())
    return {
        "workflow_id": "milestone-m1",
        "status": "completed",
        "user_question": "What is the total revenue?",
        "data_folder": folder,
        "available_files": [
            {
                "file_path": resolved,
                "file_name": csv_path.name,
                "file_type": "csv",
            }
        ],
        "route": "tabular",
        "selected_file_path": resolved,
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
        "metadata": {
            "agent_trace": [
                {"node": "init_workflow"},
                {"node": "query_planner"},
                {"node": "inspect_dataset"},
                {"node": "generate_pandas"},
                {"node": "code_executor"},
                {"node": "data_analyst"},
            ],
        },
    }


def _document_miss_state(folder: str, doc_path: Path) -> dict:
    resolved = str(doc_path.resolve())
    return {
        "workflow_id": "milestone-m3",
        "status": "completed",
        "user_question": "What is the plastics policy?",
        "data_folder": folder,
        "available_files": [
            {
                "file_path": resolved,
                "file_name": doc_path.name,
                "file_type": "document",
            }
        ],
        "route": "document",
        "selected_file_path": resolved,
        "cache_hit": False,
        "retrieval_result": {
            "llm_answer": "Plastics must be recyclable by 2026.",
            "sources": [{"chunk_id": f"{resolved}::chunk::0", "file_path": resolved}],
        },
        "analyst_output": {
            "final_answer": f"Plastics must be recyclable by 2026 ({resolved}).",
            "error_message": None,
            "confidence": "high",
        },
        "metadata": {
            "agent_trace": [
                {"node": "init_workflow"},
                {"node": "query_planner"},
                {"node": "cache_lookup"},
                {"node": "retriever"},
                {"node": "reranker"},
                {"node": "llm_answer"},
                {"node": "cache_store"},
                {"node": "data_analyst"},
            ],
        },
    }


class MilestoneSmokeTests(unittest.TestCase):
    """API milestone checks with mocked graph output (no OpenAI key required)."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_m1_tabular_ask_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            final_state = _tabular_success_state(folder, csv_path)

            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value=final_state)

            with patch("app.api.routes.get_rag_graph", return_value=mock_graph):
                response = self.client.post(
                    "/ask",
                    json={
                        "question": "What is the total revenue?",
                        "data_folder": folder,
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["route"], "tabular")
        self.assertTrue(body["execution_result"]["success"])
        self.assertEqual(body["answer"], "Total revenue is $300.")
        self.assertIsNotNone(body["workflow_id"])

    def test_m2_friendly_error_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            csv_path = Path(folder) / "sales.csv"
            final_state = {
                "workflow_id": "milestone-m2",
                "status": "completed",
                "user_question": "What is the total revenue?",
                "data_folder": folder,
                "available_files": [],
                "route": "tabular",
                "execution_result": {
                    "success": False,
                    "attempts": 2,
                    "error": "KeyError: 'Region'",
                },
                "execution_attempts": 2,
                "analyst_output": {
                    "final_answer": (
                        "I couldn't complete that query. "
                        "Available columns: region, revenue."
                    ),
                    "error_message": (
                        "KeyError: 'Region' Available columns: region, revenue."
                    ),
                    "confidence": "low",
                },
                "metadata": {
                    "agent_trace": [
                        {"node": "code_executor"},
                        {"node": "fix_pandas_query"},
                        {"node": "code_executor"},
                        {"node": "data_analyst"},
                    ],
                },
            }

            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value=final_state)

            with patch("app.api.routes.get_rag_graph", return_value=mock_graph):
                response = self.client.post(
                    "/ask",
                    json={
                        "question": "What is the total revenue?",
                        "data_folder": folder,
                    },
                )

        body = response.json()
        self.assertEqual(body["route"], "tabular")
        self.assertFalse(body["execution_result"]["success"])
        self.assertIn("Available columns: region, revenue", body["error"])
        self.assertNotIn("KeyError", body["answer"])

    def test_m3_document_path_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            doc_path = Path(folder) / "policy.txt"
            final_state = _document_miss_state(folder, doc_path)

            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(return_value=final_state)

            with patch("app.api.routes.get_rag_graph", return_value=mock_graph):
                response = self.client.post(
                    "/ask",
                    json={
                        "question": "What is the plastics policy?",
                        "data_folder": folder,
                    },
                )

        body = response.json()
        self.assertEqual(body["route"], "document")
        self.assertFalse(body["cache_hit"])
        self.assertEqual(len(body["sources"]), 1)
        self.assertIn("policy.txt", body["sources"][0]["file_path"])

    def test_m4_cache_hit_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            doc_path = Path(folder) / "policy.txt"
            miss_state = _document_miss_state(folder, doc_path)
            hit_state = {
                **miss_state,
                "workflow_id": "milestone-m4-hit",
                "cache_hit": True,
                "retrieval_result": {
                    "cached_answer": "Cached plastics policy answer.",
                    "sources": [],
                },
                "analyst_output": {
                    "final_answer": "Cached plastics policy answer.",
                    "error_message": None,
                    "confidence": "high",
                },
                "metadata": {
                    "agent_trace": [
                        {"node": "init_workflow"},
                        {"node": "query_planner"},
                        {"node": "cache_lookup"},
                        {"node": "data_analyst"},
                    ],
                },
            }

            mock_graph = AsyncMock()
            mock_graph.ainvoke = AsyncMock(
                side_effect=[miss_state, hit_state],
            )

            with patch("app.api.routes.get_rag_graph", return_value=mock_graph):
                first = self.client.post(
                    "/ask",
                    json={
                        "question": "What is the plastics policy?",
                        "data_folder": folder,
                    },
                )
                second = self.client.post(
                    "/ask",
                    json={
                        "question": "What is the plastics policy?",
                        "data_folder": folder,
                    },
                )

        self.assertFalse(first.json()["cache_hit"])
        self.assertTrue(second.json()["cache_hit"])
        self.assertEqual(
            second.json()["answer"],
            "Cached plastics policy answer.",
        )


@unittest.skipUnless(_api_key_configured(), "OPENAI_API_KEY not configured")
class MilestoneE2ETests(unittest.TestCase):
    """Live milestone validation through /ask and /workflow."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.data_folder = Path(self.temp_dir) / "company_data"
        self.data_folder.mkdir(parents=True, exist_ok=True)
        self.upload_dir = Path(self.temp_dir) / "uploads"
        self.vector_db_dir = Path(self.temp_dir) / "vector_db"
        self.cache_db_path = Path(self.temp_dir) / "cache.db"
        self.workflow_db_path = Path(self.temp_dir) / "workflows.db"

        self.settings = Settings(
            upload_dir=self.upload_dir,
            vector_db_dir=self.vector_db_dir,
            cache_db_path=self.cache_db_path,
            workflow_db_path=self.workflow_db_path,
            data_folder=self.data_folder,
        )
        get_settings.cache_clear()
        self.settings_patch = patch(
            "app.config.get_settings",
            return_value=self.settings,
        )
        self.routes_settings_patch = patch(
            "app.api.routes.get_settings",
            return_value=self.settings,
        )
        self.settings_patch.start()
        self.routes_settings_patch.start()
        self._client_context = TestClient(app)
        self.client = self._client_context.__enter__()

    def tearDown(self) -> None:
        self._client_context.__exit__(None, None, None)
        get_settings.cache_clear()
        self.routes_settings_patch.stop()
        self.settings_patch.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_m1_tabular_csv_ask_e2e(self) -> None:
        csv_path = self.data_folder / "sales.csv"
        csv_path.write_text(
            "region,revenue\nNorth,100\nSouth,200\nEast,50\n",
            encoding="utf-8",
        )

        response = self.client.post(
            "/ask",
            json={
                "question": "What is the total revenue across all regions?",
                "data_folder": str(self.data_folder),
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["route"], "tabular")
        self.assertTrue(body["execution_result"]["success"])
        self.assertIsNotNone(body["answer"])

        workflow = self.client.get(f"/workflow/{body['workflow_id']}")
        self.assertEqual(workflow.status_code, 200)
        visited = [
            entry["node"]
            for entry in workflow.json()["state"]["metadata"]["agent_trace"]
        ]
        self.assertIn("code_executor", visited)
        self.assertIn("data_analyst", visited)

    @patch("app.graph.workflow.fix_pandas_query_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.generate_pandas_node", new_callable=AsyncMock)
    @patch("app.graph.workflow.query_planner_node", new_callable=AsyncMock)
    def test_m2_bad_column_retry_friendly_error_e2e(
        self,
        mock_query_planner: AsyncMock,
        mock_generate_pandas: AsyncMock,
        mock_fix_pandas_query: AsyncMock,
    ) -> None:
        csv_path = self.data_folder / "sales.csv"
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

        graph = _compile_checkpointed_graph()
        with patch("app.api.routes.get_rag_graph", return_value=graph):
            response = self.client.post(
                "/ask",
                json={
                    "question": "What is the total revenue across all regions?",
                    "data_folder": str(self.data_folder),
                },
            )

        body = response.json()
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["route"], "tabular")
        self.assertFalse(body["execution_result"]["success"])
        self.assertIn("fix_pandas_query", _visited_nodes(
            self.client.get(f"/workflow/{body['workflow_id']}").json()["state"]
        ))
        self.assertIn("Available columns: region, revenue", body["error"])
        self.assertNotIn("KeyError", body["answer"])

    def test_m3_upload_ingest_ask_document_e2e(self) -> None:
        notes_path = self.data_folder / "policy.txt"
        notes_path.write_text(
            "All plastics packaging must be recyclable by 2026.",
            encoding="utf-8",
        )

        async def fake_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

        fake_chunks = [
            {
                "chunk_id": f"{notes_path.resolve()}::chunk::0",
                "text": "All plastics packaging must be recyclable by 2026.",
                "score": 0.95,
                "file_path": str(notes_path.resolve()),
                "chunk_index": 0,
            }
        ]

        with (
            patch("app.rag.ingest.get_settings", return_value=self.settings),
            patch("app.rag.cache.get_settings", return_value=self.settings),
            patch("app.rag.ingest.embed_texts", new=fake_embed),
            patch(
                "app.graph.nodes.retriever.retrieve_chunks",
                return_value=fake_chunks,
            ),
            patch(
                "app.graph.nodes.llm_answer.call_llm",
                new_callable=AsyncMock,
                return_value="Plastics must be recyclable by 2026.",
            ),
        ):
            ingest_response = self.client.post(
                "/ingest",
                json={"file_path": str(notes_path)},
            )
            self.assertEqual(ingest_response.status_code, 200)
            self.assertGreater(ingest_response.json()["chunks_indexed"], 0)

            ask_response = self.client.post(
                "/ask",
                json={
                    "question": "What is the plastics policy?",
                    "data_folder": str(self.data_folder),
                },
            )

        body = ask_response.json()
        self.assertEqual(ask_response.status_code, 200)
        self.assertEqual(body["route"], "document")
        self.assertFalse(body["cache_hit"])
        self.assertGreaterEqual(len(body["sources"]), 1)
        self.assertIsNotNone(body["answer"])

        visited = [
            entry["node"]
            for entry in self.client.get(f"/workflow/{body['workflow_id']}")
            .json()["state"]["metadata"]["agent_trace"]
        ]
        self.assertIn("cache_lookup", visited)
        self.assertIn("retriever", visited)
        self.assertIn("llm_answer", visited)
        self.assertIn("cache_store", visited)
        self.assertIn("data_analyst", visited)

    def test_m4_repeat_question_cache_hit_e2e(self) -> None:
        notes_path = self.data_folder / "policy.txt"
        notes_path.write_text(
            "All plastics packaging must be recyclable by 2026.",
            encoding="utf-8",
        )

        async def fake_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

        fake_chunks = [
            {
                "chunk_id": f"{notes_path.resolve()}::chunk::0",
                "text": "All plastics packaging must be recyclable by 2026.",
                "score": 0.95,
                "file_path": str(notes_path.resolve()),
                "chunk_index": 0,
            }
        ]

        question = "What is the plastics policy?"

        with (
            patch("app.rag.ingest.get_settings", return_value=self.settings),
            patch("app.rag.cache.get_settings", return_value=self.settings),
            patch("app.rag.ingest.embed_texts", new=fake_embed),
            patch(
                "app.graph.nodes.retriever.retrieve_chunks",
                return_value=fake_chunks,
            ),
            patch(
                "app.graph.nodes.llm_answer.call_llm",
                new_callable=AsyncMock,
                return_value="Plastics must be recyclable by 2026.",
            ),
        ):
            self.client.post("/ingest", json={"file_path": str(notes_path)})

            first = self.client.post(
                "/ask",
                json={"question": question, "data_folder": str(self.data_folder)},
            )
            second = self.client.post(
                "/ask",
                json={"question": question, "data_folder": str(self.data_folder)},
            )

        first_body = first.json()
        second_body = second.json()
        self.assertFalse(first_body["cache_hit"])
        self.assertTrue(second_body["cache_hit"])

        second_trace = [
            entry["node"]
            for entry in self.client.get(f"/workflow/{second_body['workflow_id']}")
            .json()["state"]["metadata"]["agent_trace"]
        ]
        self.assertIn("cache_lookup", second_trace)
        self.assertNotIn("retriever", second_trace)
        self.assertNotIn("llm_answer", second_trace)

    def test_ask_handler_m1_live(self) -> None:
        csv_path = self.data_folder / "sales.csv"
        csv_path.write_text("region,revenue\nNorth,100\nSouth,200\n", encoding="utf-8")

        response = asyncio.run(
            ask(
                AskRequest(
                    question="What is the total revenue across all regions?",
                    data_folder=str(self.data_folder),
                )
            )
        )

        self.assertEqual(response.status, "completed")
        self.assertEqual(response.route, "tabular")
        self.assertTrue(response.execution_result.success)


if __name__ == "__main__":
    unittest.main()
