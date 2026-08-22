from __future__ import annotations

from pathlib import Path

import fitz
from docx import Document

from app.utils.file_utils import detect_file_type, validate_local_file


def load_document_text(file_path: str | Path) -> str:
    """Extract plain text from a local PDF, DOCX, or TXT file."""
    path = validate_local_file(file_path)
    file_type = detect_file_type(path)
    if file_type != "document":
        raise ValueError(
            f"Only document files can be ingested (PDF, DOCX, TXT), got '{file_type}'."
        )

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf_text(path)
    if suffix == ".docx":
        return _load_docx_text(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    raise ValueError(f"Unsupported document extension: {suffix}")


def _load_pdf_text(path: Path) -> str:
    with fitz.open(path) as doc:
        pages = [page.get_text() for page in doc]
    return "\n".join(page for page in pages if page.strip())


def _load_docx_text(path: Path) -> str:
    document = Document(path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)
