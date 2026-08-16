from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph.state import Route, WorkflowState
from app.llm.openai_client import call_llm_json
from app.rag.prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    build_query_planner_user_prompt,
)

_TABULAR_TYPES = frozenset({"csv", "excel"})


def _append_trace(state: WorkflowState, node: str, **extra: Any) -> list[dict[str, Any]]:
    metadata = state.get("metadata") or {}
    trace = list(metadata.get("agent_trace", []))
    trace.append(
        {
            "node": node,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
    )
    return trace


def _index_available_files(
    available_files: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for file_info in available_files:
        indexed[str(Path(file_info["file_path"]).resolve())] = file_info
        indexed[file_info["file_path"]] = file_info
        indexed[file_info["file_name"]] = file_info
    return indexed


def _resolve_selected_file(
    selected_file_path: str,
    available_files: list[dict[str, str]],
) -> dict[str, str]:
    indexed = _index_available_files(available_files)

    candidates = [
        selected_file_path,
        str(Path(selected_file_path).resolve()),
        Path(selected_file_path).name,
    ]
    for candidate in candidates:
        if candidate in indexed:
            return indexed[candidate]

    raise ValueError(
        f"LLM selected unknown file: {selected_file_path!r}. "
        f"Must be one of: {[f['file_path'] for f in available_files]}"
    )


def _expected_route(file_type: str) -> Route:
    if file_type in _TABULAR_TYPES:
        return "tabular"
    if file_type == "document":
        return "document"
    raise ValueError(f"Unsupported file type for routing: {file_type}")


def _parse_planner_response(
    response: dict[str, Any],
    available_files: list[dict[str, str]],
    user_question: str,
) -> tuple[Route, dict[str, str], str, str | None]:
    route = response.get("route")
    if route not in ("tabular", "document"):
        raise ValueError(f'LLM returned invalid route: {route!r}')

    reasoning = response.get("reasoning")
    if not reasoning or not isinstance(reasoning, str):
        raise ValueError("LLM response missing non-empty 'reasoning'.")

    selected_file_path = response.get("selected_file_path")
    if not selected_file_path or not isinstance(selected_file_path, str):
        raise ValueError("LLM response missing 'selected_file_path'.")

    selected_file = _resolve_selected_file(selected_file_path, available_files)
    expected_route = _expected_route(selected_file["file_type"])
    if route != expected_route:
        raise ValueError(
            f"LLM route {route!r} does not match selected file type "
            f"{selected_file['file_type']!r}."
        )

    retrieval_query = response.get("retrieval_query")
    if route == "tabular":
        retrieval_query = None
    elif not retrieval_query:
        retrieval_query = user_question

    return route, selected_file, reasoning, retrieval_query


async def query_planner_node(state: WorkflowState) -> dict[str, Any]:
    """
    Agent 1 — LLM decides route, picks a file, and sets retrieval_query.
    """
    available_files = state.get("available_files") or []
    if not available_files:
        raise ValueError("query_planner requires available_files from init_workflow.")

    user_question = state["user_question"]
    user_prompt = build_query_planner_user_prompt(user_question, available_files)
    llm_response = await call_llm_json(
        QUERY_PLANNER_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
    )

    route, selected_file, reasoning, retrieval_query = _parse_planner_response(
        llm_response,
        available_files,
        user_question,
    )
    file_type = selected_file["file_type"]
    if file_type not in (*_TABULAR_TYPES, "document"):
        raise ValueError(f"Unsupported file type for routing: {file_type}")

    planner_output: dict[str, Any] = {
        "route": route,
        "reasoning": reasoning,
        "dataset_summary": None,
        "pandas_query": None,
        "retrieval_query": retrieval_query,
        "queries": [],
    }

    return {
        "status": "planning",
        "route": route,
        "selected_file_path": selected_file["file_path"],
        "selected_file_type": file_type,  # type: ignore[typeddict-item]
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "query_planner",
                route=route,
                selected_file=selected_file["file_name"],
            ),
        },
    }
