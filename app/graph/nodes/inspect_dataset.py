from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState


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
    """Tabular-path stub — placeholder dataset summary until Phase 3."""
    planner_output = dict(state.get("planner_output") or {})
    planner_output["dataset_summary"] = {
        "status": "stub",
        "message": "Dataset inspection not implemented yet (Phase 3).",
        "file_path": state.get("selected_file_path"),
    }

    return {
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "inspect_dataset",
                file_path=state.get("selected_file_path"),
            ),
        },
    }
