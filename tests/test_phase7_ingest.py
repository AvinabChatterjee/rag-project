import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.rag.ingest import chunk_text, ingest_document


class Phase7IngestEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_ingest_returns_resolved_path_and_chunk_count(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write("Hello from a document for ingest testing.")
            path = Path(handle.name)

        try:
            with patch(
                "app.api.routes.ingest_document",
                new_callable=AsyncMock,
                return_value=5,
            ) as mock_ingest:
                response = self.client.post(
                    "/ingest",
                    json={"file_path": str(path)},
                )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["chunks_indexed"], 5)
            self.assertEqual(body["file_path"], str(path.resolve()))
            mock_ingest.assert_awaited_once_with(path.resolve())
        finally:
            path.unlink(missing_ok=True)

    def test_ingest_validates_file_before_pipeline(self) -> None:
        with patch(
            "app.api.routes.ingest_document",
            new_callable=AsyncMock,
        ) as mock_ingest:
            response = self.client.post(
                "/ingest",
                json={"file_path": str(Path(tempfile.gettempdir()) / "missing-doc.txt")},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("File not found", response.json()["detail"])
        mock_ingest.assert_not_called()

    def test_ingest_rejects_csv(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write("region,revenue\nNorth,100\n")
            path = Path(handle.name)

        try:
            response = self.client.post(
                "/ingest",
                json={"file_path": str(path)},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("Only document files can be ingested", response.json()["detail"])
        finally:
            path.unlink(missing_ok=True)

    def test_ingest_rejects_blank_file_path(self) -> None:
        response = self.client.post("/ingest", json={"file_path": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("file_path is required", response.json()["detail"])

    def test_ingest_rejects_empty_file_path(self) -> None:
        response = self.client.post("/ingest", json={"file_path": ""})
        self.assertEqual(response.status_code, 422)


class Phase7IngestIntegrationTests(unittest.TestCase):
    def test_ingest_indexes_txt_into_chroma(self) -> None:
        temp_root = tempfile.mkdtemp()
        vector_dir = Path(temp_root) / "vector_db"
        try:
            path = Path(temp_root) / "notes.txt"
            path.write_text(
                " ".join(f"word{i}" for i in range(80)),
                encoding="utf-8",
            )
            expected_chunks = chunk_text(
                path.read_text(encoding="utf-8"),
                chunk_size=20,
                chunk_overlap=5,
            )

            async def fake_embed(texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

            with (
                patch("app.rag.ingest.get_settings") as mock_settings,
                patch("app.rag.ingest.embed_texts", new=fake_embed),
                patch("app.rag.ingest.chunk_text", return_value=expected_chunks),
                patch("app.rag.ingest.get_chroma_collection") as mock_get_collection,
            ):
                settings = mock_settings.return_value
                settings.vector_db_dir = vector_dir
                settings.chroma_collection_name = "test_documents"

                mock_collection = mock_get_collection.return_value
                mock_collection.get.return_value = {"ids": []}

                chunks_indexed = asyncio.run(ingest_document(path))

            self.assertEqual(chunks_indexed, len(expected_chunks))
            mock_collection.add.assert_called_once()
            add_kwargs = mock_collection.add.call_args.kwargs
            self.assertEqual(
                add_kwargs["metadatas"][0]["file_path"],
                str(path.resolve()),
            )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
