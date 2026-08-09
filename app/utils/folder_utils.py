from pathlib import Path
from typing import TypedDict

from app.utils.file_utils import ALLOWED_EXTENSIONS, detect_file_type


class ScannedFile(TypedDict):
    file_path: str
    file_name: str
    file_type: str


def scan_data_folder(folder_path: str | Path) -> list[ScannedFile]:
    """
    List supported data files in a folder (top level only).

    Returns metadata for each file Agent 1 will use to pick the right source.
    """
    folder = Path(folder_path).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Data folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Data folder path is not a directory: {folder}")

    scanned: list[ScannedFile] = []
    for entry in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.name.startswith("."):
            continue
        if entry.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        scanned.append(
            {
                "file_path": str(entry.resolve()),
                "file_name": entry.name,
                "file_type": detect_file_type(entry),
            }
        )

    return scanned
