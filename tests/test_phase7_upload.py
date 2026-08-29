import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.utils.file_utils import detect_file_type, save_upload


class Phase7FileUtilsTests(unittest.TestCase):
    def test_detect_file_type_csv(self) -> None:
        self.assertEqual(detect_file_type("sales.csv"), "csv")

    def test_detect_file_type_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            detect_file_type("data.json")

    def test_save_upload_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            upload_dir = Path(folder)
            saved = save_upload(b"region,revenue\nNorth,100", "sales.csv", upload_dir)
            self.assertTrue(saved.exists())
            self.assertEqual(saved.read_bytes(), b"region,revenue\nNorth,100")
            self.assertTrue(saved.name.endswith("_sales.csv"))

    def test_save_upload_rejects_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                save_upload(b"", "sales.csv", Path(folder))

    def test_save_upload_strips_path_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            upload_dir = Path(folder)
            saved = save_upload(b"ok", "../../evil.csv", upload_dir)
            self.assertTrue(saved.name.endswith("_evil.csv"))
            self.assertEqual(saved.parent.resolve(), upload_dir.resolve())


class Phase7UploadEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.upload_dir = Path(self.temp_dir) / "uploads"
        settings = Settings()
        settings.upload_dir = self.upload_dir
        settings.ensure_directories()
        self.settings_patch = patch(
            "app.api.routes.get_settings",
            return_value=settings,
        )
        self.settings_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.settings_patch.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upload_csv_returns_path_and_type(self) -> None:
        response = self.client.post(
            "/upload",
            files={"file": ("sales.csv", "region,revenue\nNorth,100", "text/csv")},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["file_type"], "csv")
        self.assertEqual(body["filename"], Path(body["file_path"]).name)
        self.assertTrue(Path(body["file_path"]).exists())

    def test_upload_excel_extension_maps_to_excel(self) -> None:
        response = self.client.post(
            "/upload",
            files={"file": ("report.xlsx", b"not-a-real-xlsx", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["file_type"], "excel")

    def test_upload_pdf_maps_to_document(self) -> None:
        response = self.client.post(
            "/upload",
            files={"file": ("notes.pdf", b"%PDF-1.4", "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["file_type"], "document")

    def test_upload_rejects_unsupported_extension(self) -> None:
        response = self.client.post(
            "/upload",
            files={"file": ("data.json", "{}", "application/json")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file type", response.json()["detail"])

    def test_upload_rejects_empty_file(self) -> None:
        response = self.client.post(
            "/upload",
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_upload_rejects_blank_filename(self) -> None:
        response = self.client.post(
            "/upload",
            files={"file": ("   ", b"data", "text/csv")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Filename is required", response.json()["detail"])
