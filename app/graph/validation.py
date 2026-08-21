"""Validation helpers for Agent 1 (Query Planner) and Agent 2 nodes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.graph.state import Route

_TABULAR_TYPES = frozenset({"csv", "excel"})
_FORBIDDEN_PANDAS_PATTERNS = (
    "read_csv",
    "read_excel",
    "open(",
    "import ",
    "exec(",
    "eval(",
)
_DF_COL_BRACKET = re.compile(r"""df\[['"]([^'"]+)['"]\]""")
_DF_COL_DOT = re.compile(r"df\.([A-Za-z_]\w*)")


def index_available_files(
    available_files: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for file_info in available_files:
        indexed[str(Path(file_info["file_path"]).resolve())] = file_info
        indexed[file_info["file_path"]] = file_info
        indexed[file_info["file_name"]] = file_info
    return indexed


def resolve_selected_file(
    selected_file_path: str,
    available_files: list[dict[str, str]],
) -> dict[str, str]:
    indexed = index_available_files(available_files)
    candidates = [
        selected_file_path,
        str(Path(selected_file_path).resolve()),
        Path(selected_file_path).name,
    ]
    for candidate in candidates:
        if candidate in indexed:
            return indexed[candidate]

    available_paths = [file_info["file_path"] for file_info in available_files]
    raise ValueError(
        f"LLM selected unknown file: {selected_file_path!r}. "
        f"Must be one of: {available_paths}"
    )


def expected_route(file_type: str) -> Route:
    if file_type in _TABULAR_TYPES:
        return "tabular"
    if file_type == "document":
        return "document"
    raise ValueError(f"Unsupported file type for routing: {file_type!r}")


def parse_planner_response(
    response: dict[str, Any],
    available_files: list[dict[str, str]],
    user_question: str,
) -> tuple[Route, dict[str, str], str, str | None]:
    if not isinstance(response, dict):
        raise ValueError("LLM planner response must be a JSON object.")

    route = response.get("route")
    if route not in ("tabular", "document"):
        raise ValueError(
            f'LLM returned invalid route: {route!r}. Expected "tabular" or "document".'
        )

    reasoning = response.get("reasoning")
    if not reasoning or not isinstance(reasoning, str):
        raise ValueError("LLM planner response missing non-empty 'reasoning'.")

    selected_file_path = response.get("selected_file_path")
    if not selected_file_path or not isinstance(selected_file_path, str):
        raise ValueError("LLM planner response missing 'selected_file_path'.")

    selected_file = resolve_selected_file(selected_file_path, available_files)
    if route != expected_route(selected_file["file_type"]):
        raise ValueError(
            f"LLM route {route!r} does not match selected file type "
            f"{selected_file['file_type']!r}."
        )

    retrieval_query = response.get("retrieval_query")
    if route == "tabular":
        retrieval_query = None
    elif not retrieval_query or not isinstance(retrieval_query, str):
        retrieval_query = user_question

    return route, selected_file, reasoning, retrieval_query


def get_dataset_columns(dataset_summary: dict[str, Any]) -> set[str]:
    dtypes = dataset_summary.get("dtypes") or {}
    return {str(column) for column in dtypes.keys()}


def _referenced_columns(query: str) -> set[str]:
    columns = set(_DF_COL_BRACKET.findall(query))
    columns.update(_DF_COL_DOT.findall(query))
    return columns


def validate_pandas_query(query: str, dataset_summary: dict[str, Any]) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("LLM response missing non-empty 'pandas_query'.")

    normalized = query.strip()
    lowered = normalized.lower()
    for pattern in _FORBIDDEN_PANDAS_PATTERNS:
        if pattern in lowered:
            raise ValueError(f"Generated Pandas query must not contain '{pattern}'.")

    columns = get_dataset_columns(dataset_summary)
    if columns:
        unknown = _referenced_columns(normalized) - columns
        if unknown:
            raise ValueError(
                f"Generated Pandas query uses unknown columns: {sorted(unknown)}. "
                f"Available columns: {sorted(columns)}"
            )

    return normalized


def parse_pandas_response(
    response: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> str:
    if not isinstance(response, dict):
        raise ValueError("LLM pandas response must be a JSON object.")
    return validate_pandas_query(response.get("pandas_query", ""), dataset_summary)


def require_successful_dataset_summary(dataset_summary: dict[str, Any] | None) -> dict[str, Any]:
    if not dataset_summary:
        raise ValueError("Cannot generate Pandas query: dataset summary is missing.")
    if dataset_summary.get("status") != "success":
        error = dataset_summary.get("error", "Dataset inspection failed.")
        raise ValueError(f"Cannot generate Pandas query: {error}")
    return dataset_summary


def require_selected_file_path(selected_file_path: str | None) -> str:
    if not selected_file_path:
        raise ValueError("code_executor requires selected_file_path from query_planner.")
    return selected_file_path


def require_pandas_query(planner_output: dict[str, Any] | None) -> str:
    planner_output = planner_output or {}
    pandas_query = planner_output.get("pandas_query")
    if not pandas_query or not str(pandas_query).strip():
        raise ValueError("code_executor requires planner_output.pandas_query.")
    return str(pandas_query).strip()


def require_failed_execution(execution_result: dict[str, Any] | None) -> str:
    execution_result = execution_result or {}
    if execution_result.get("success"):
        raise ValueError("fix_pandas_query requires a failed execution_result.")
    error_message = execution_result.get("error")
    if not error_message:
        raise ValueError("fix_pandas_query requires execution_result.error.")
    return str(error_message)


def build_execution_error_message(
    execution_error: str,
    dataset_summary: dict[str, Any] | None,
) -> str:
    columns = sorted(get_dataset_columns(dataset_summary or {}))
    if columns:
        return f"{execution_error} Available columns: {', '.join(columns)}."
    return execution_error
