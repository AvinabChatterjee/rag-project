from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from app.utils.file_utils import detect_file_type, validate_local_file

_MAX_UNIQUE_VALUES = 50
_MAX_DATAFRAME_ROWS = 100
_FILE_LOADING_MARKERS = ("read_csv", "read_excel")


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
    if isinstance(value, pd.DataFrame):
        truncated = value.head(_MAX_DATAFRAME_ROWS)
        return truncated.to_dict(orient="records")

    if isinstance(value, pd.Series):
        return {str(key): _to_serializable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _strip_file_loading_lines(query: str) -> str:
    kept_lines: list[str] = []
    for line in query.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lowered = stripped.lower()
        if any(marker in lowered for marker in _FILE_LOADING_MARKERS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


def _execute_pandas_query(df: pd.DataFrame, query: str) -> Any:
    cleaned_query = _strip_file_loading_lines(query)
    if not cleaned_query:
        raise ValueError("Pandas query is empty after removing file-loading lines.")

    scope: dict[str, Any] = {
        "df": df,
        "pandas": pd,
        "pd": pd,
        "__builtins__": __builtins__,
    }
    lines = [line.strip() for line in cleaned_query.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Pandas query has no executable lines.")

    if len(lines) > 1:
        exec("\n".join(lines[:-1]), scope)  # noqa: S102

    last_line = lines[-1]
    try:
        return eval(last_line, scope)  # noqa: S307
    except SyntaxError:
        exec(last_line, scope)  # noqa: S102
        return None


def run_query_data(
    file_path: str,
    query: str,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """
    Execute generated Pandas code against a local CSV/Excel file.

    The DataFrame is pre-loaded as `df` with normalized column names.
    """
    try:
        df = _normalize_columns(_load_local_dataframe(file_path, sheet_name))
        result = _execute_pandas_query(df, query)
        cleaned_query = _strip_file_loading_lines(query)
        return {
            "status": "success",
            "result": _to_serializable(result),
            "executed_code": cleaned_query,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "executed_code": _strip_file_loading_lines(query),
        }


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
