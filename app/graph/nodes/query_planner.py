from datetime import datetime, timezone
from typing import Any

from app.graph.state import Route, WorkflowState

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


def _pick_route_and_file(
    available_files: list[dict[str, str]],
) -> tuple[Route, dict[str, str]]:
    for file_info in available_files:
        if file_info["file_type"] in _TABULAR_TYPES:
            return "tabular", file_info

    for file_info in available_files:
        if file_info["file_type"] == "document":
            return "document", file_info

    raise ValueError("No tabular or document files available for routing.")


def query_planner_node(state: WorkflowState) -> dict[str, Any]:
    """
    Agent 1 stub — picks a route and target file without calling the LLM.

    Prefers the first tabular file; otherwise uses the first document file.
    """
    available_files = state.get("available_files") or []
    if not available_files:
        raise ValueError("query_planner requires available_files from init_workflow.")

    route, selected_file = _pick_route_and_file(available_files)
    file_type = selected_file["file_type"]
    if file_type not in (*_TABULAR_TYPES, "document"):
        raise ValueError(f"Unsupported file type for routing: {file_type}")

    planner_output: dict[str, Any] = {
        "route": route,
        "reasoning": f"Phase 2 stub: selected {selected_file['file_name']} for {route} path.",
        "dataset_summary": None,
        "pandas_query": None,
        "retrieval_query": state["user_question"] if route == "document" else None,
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
