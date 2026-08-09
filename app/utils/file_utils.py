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


def save_upload(file_bytes: bytes, original_filename: str, upload_dir: Path) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)

    source = Path(original_filename)
    suffix = source.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    stem = source.stem.replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination = upload_dir / f"{timestamp}_{stem}{suffix}"
    destination.write_bytes(file_bytes)
    return destination.resolve()
