from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.file_utils import detect_file_type, validate_local_file

_MAX_UNIQUE_VALUES = 50


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\w]", "_", regex=True)
    )
    return df


def _load_local_dataframe(file_path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    path = validate_local_file(file_path)
    file_type = detect_file_type(path)

    if file_type == "csv":
        return pd.read_csv(path)

    if file_type == "excel":
        return pd.read_excel(path, sheet_name=sheet_name or 0)

    raise ValueError(
        f"get_dataframe_info only supports CSV/Excel files, got '{file_type}'."
    )


def _to_serializable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def get_dataframe_info(file_path: str, sheet_name: str | None = None) -> dict[str, Any]:
    """
    Inspect a local CSV/Excel file for Agent 1 (Query Planner).

    Returns column dtypes, capped unique values, and sample rows.
    """
    try:
        df = _load_local_dataframe(file_path, sheet_name)
        df = _normalize_columns(df)

        unique_values: dict[str, list[Any]] = {}
        for col in df.columns:
            vals = [_to_serializable(v) for v in df[col].dropna().unique().tolist()]
            if len(vals) > _MAX_UNIQUE_VALUES:
                unique_values[col] = vals[:_MAX_UNIQUE_VALUES] + [
                    f"[... {len(vals) - _MAX_UNIQUE_VALUES} more values truncated]"
                ]
            else:
                unique_values[col] = vals

        return {
            "status": "success",
            "shape": list(df.shape),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "unique_values": unique_values,
            "sample_rows": df.head(3).to_string(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
        }
