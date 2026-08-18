from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.tools.dataframe_tools import get_dataframe_info

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


def inspect_dataset_node(state: WorkflowState) -> dict[str, Any]:
    """Tabular path — inspect the selected CSV/Excel file for Agent 1."""
    selected_file_path = state.get("selected_file_path")
    if not selected_file_path:
        raise ValueError("inspect_dataset requires selected_file_path from query_planner.")

    selected_file_type = state.get("selected_file_type")
    if selected_file_type not in _TABULAR_TYPES:
        raise ValueError(
            f"inspect_dataset only runs for tabular files, got {selected_file_type!r}."
        )

    dataset_summary = get_dataframe_info(selected_file_path)
    planner_output = dict(state.get("planner_output") or {})
    planner_output["dataset_summary"] = dataset_summary

    trace_extra: dict[str, Any] = {"file_path": selected_file_path}
    status = "planning"
    if dataset_summary.get("status") == "success":
        trace_extra["shape"] = dataset_summary.get("shape")
    else:
        trace_extra["error"] = dataset_summary.get("error")
        status = "failed"

    return {
        "status": status,
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(state, "inspect_dataset", **trace_extra),
        },
    }
