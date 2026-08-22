import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.rag.ingest import chunk_text, ingest_document
from app.tools.document_tools import load_document_text


class Phase5IngestTests(unittest.TestCase):
    def test_load_txt_document(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write("Hello from a plain text document.")
            path = Path(handle.name)

        try:
            text = load_document_text(path)
        finally:
            path.unlink(missing_ok=True)

        self.assertIn("plain text document", text)

    def test_chunk_text_respects_overlap(self) -> None:
        text = " ".join(f"word{i}" for i in range(2000))
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_ingest_document_rejects_csv(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".csv",
            delete=False,
            encoding="utf-8",
        ) as handle:
            handle.write("region,revenue\nNorth,100\n")
            path = Path(handle.name)

        try:
            with self.assertRaises(ValueError):
                asyncio.run(ingest_document(path))
        finally:
            path.unlink(missing_ok=True)

    def test_ingest_document_indexes_txt(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "notes.txt"
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
                patch("app.rag.ingest.embed_texts", new=fake_embed),
                patch("app.rag.ingest.chunk_text", return_value=expected_chunks),
                patch("app.rag.ingest.get_chroma_collection") as mock_get_collection,
            ):
                mock_collection = mock_get_collection.return_value
                mock_collection.get.return_value = {"ids": []}

                chunks_indexed = asyncio.run(ingest_document(path))

            self.assertEqual(chunks_indexed, len(expected_chunks))
            mock_collection.add.assert_called_once()
            add_kwargs = mock_collection.add.call_args.kwargs
            self.assertEqual(len(add_kwargs["documents"]), len(expected_chunks))
            self.assertEqual(
                add_kwargs["metadatas"][0]["file_path"],
                str(path.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
