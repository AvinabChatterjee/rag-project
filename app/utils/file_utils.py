from datetime import datetime, timezone
from pathlib import Path

ALLOWED_EXTENSIONS = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".pdf": "document",
    ".txt": "document",
    ".docx": "document",
}


def detect_file_type(file_path: str | Path) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ALLOWED_EXTENSIONS[suffix]


def validate_local_file(file_path: str | Path) -> Path:
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    detect_file_type(path)
    return path


def _safe_upload_stem(original_filename: str) -> str:
    """Strip path segments and unsafe characters from the upload filename stem."""
    name = Path(original_filename).name
    stem = Path(name).stem.replace(" ", "_")
    for char in ("/", "\\", "\0"):
        stem = stem.replace(char, "_")
    return stem or "upload"


def save_upload(file_bytes: bytes, original_filename: str, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    source = Path(original_filename).name
    suffix = Path(source).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    stem = _safe_upload_stem(original_filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = upload_dir / f"{timestamp}_{stem}{suffix}"
    destination.write_bytes(file_bytes)
    return destination.resolve()
