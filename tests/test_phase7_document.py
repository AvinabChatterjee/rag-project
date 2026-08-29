import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.api.routes import AskRequest, ask, build_ask_response
from app.main import app


def _document_final_state(
    folder: str,
    pdf_path: Path,
    *,
    cache_hit: bool = False,
) -> dict:
    resolved = str(pdf_path.resolve())
    sources = [{"chunk_id": "chunk-1", "file_path": resolved}]
    return {
        "workflow_id": "wf-doc-123",
        "status": "completed",
        "user_question": "What is the plastics policy?",
        "data_folder": folder,
        "available_files": [
            {
                "file_path": resolved,
                "file_name": pdf_path.name,
                "file_type": "document",
            }
        ],
        "route": "document",
        "selected_file_path": resolved,
        "cache_hit": cache_hit,
        "retrieval_result": {
            "cached_answer": "Cached plastics policy answer." if cache_hit else None,
            "llm_answer": None if cache_hit else "Plastics must be recyclable by 2026.",
            "sources": sources if not cache_hit else [],
        },
        "analyst_output": {
            "final_answer": (
                f"Plastics must be recyclable by 2026 ({resolved})."
            ),
            "error_message": None,
            "confidence": "high",
        },
    }


class Phase7DocumentMappingTests(unittest.TestCase):
    def test_build_ask_response_maps_document_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "policy.pdf"
            final_state = _document_final_state(folder, pdf_path)

            response = build_ask_response(final_state)

        self.assertEqual(response.route, "document")
        self.assertEqual(response.cache_hit, False)
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(response.sources[0].chunk_id, "chunk-1")
        self.assertIn("policy.pdf", response.sources[0].file_path or "")

    def test_build_ask_response_maps_cache_hit_without_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "policy.pdf"
            final_state = _document_final_state(folder, pdf_path, cache_hit=True)

            response = build_ask_response(final_state)

        self.assertTrue(response.cache_hit)
        self.assertEqual(response.sources, [])


class Phase7DocumentEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_ask_maps_document_workflow_with_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "policy.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            final_state = _document_final_state(folder, pdf_path)

            with patch(
                "app.api.routes.rag_graph.ainvoke",
                new_callable=AsyncMock,
                return_value=final_state,
            ):
                response = self.client.post(
                    "/ask",
                    json={
                        "question": "What is the plastics policy?",
                        "data_folder": folder,
                    },
                )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["route"], "document")
        self.assertFalse(body["cache_hit"])
        self.assertEqual(len(body["sources"]), 1)
        self.assertIn("policy.pdf", body["sources"][0]["file_path"])
        self.assertIn("policy.pdf", body["answer"])

    def test_ask_returns_400_when_document_not_ingested(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "policy.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            response = self.client.post(
                "/ask",
                json={
                    "question": "What is the plastics policy?",
                    "data_folder": folder,
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("POST /ingest", response.json()["detail"])

    def test_ask_document_handler_returns_sources(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            pdf_path = Path(folder) / "policy.pdf"
            final_state = _document_final_state(folder, pdf_path)

            with patch(
                "app.api.routes.rag_graph.ainvoke",
                new_callable=AsyncMock,
                return_value=final_state,
            ):
                response = asyncio.run(
                    ask(
                        AskRequest(
                            question="What is the plastics policy?",
                            data_folder=folder,
                        )
                    )
                )

        self.assertEqual(response.route, "document")
        self.assertEqual(len(response.sources), 1)


if __name__ == "__main__":
    unittest.main()
