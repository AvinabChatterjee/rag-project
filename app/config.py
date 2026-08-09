from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    data_dir: Path = PROJECT_ROOT / "data"
    upload_dir: Path = PROJECT_ROOT / "data" / "uploads"
    vector_db_dir: Path = PROJECT_ROOT / "data" / "vector_db"
    cache_db_path: Path = PROJECT_ROOT / "data" / "cache.db"

    # Default folder scanned for CSV/Excel/PDF when /ask omits data_folder (Option A).
    data_folder: Path | None = None

    semantic_cache_threshold: float = 0.92
    semantic_cache_ttl_hours: int = 24

    def default_data_folder(self) -> Path:
        return (self.data_dir / "company_data").resolve()

    def resolve_data_folder(self, override: str | Path | None = None) -> Path:
        if override is not None:
            folder = Path(override)
        elif self.data_folder is not None:
            folder = self.data_folder
        else:
            folder = self.default_data_folder()

        folder = folder.expanduser().resolve()
        if not folder.exists():
            raise FileNotFoundError(f"Data folder not found: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Data folder path is not a directory: {folder}")
        return folder

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)
        self.default_data_folder().mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
