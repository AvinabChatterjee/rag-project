from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState

_STUB_PANDAS_QUERY = "df.head()  # Phase 2 stub query"


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


def generate_pandas_node(state: WorkflowState) -> dict[str, Any]:
    """Tabular-path stub — placeholder Pandas query until Phase 3."""
    planner_output = dict(state.get("planner_output") or {})
    planner_output["pandas_query"] = _STUB_PANDAS_QUERY
    planner_output["queries"] = list(planner_output.get("queries") or [])
    planner_output["queries"].append(_STUB_PANDAS_QUERY)

    return {
        "status": "executing",
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(state, "generate_pandas"),
        },
    }
