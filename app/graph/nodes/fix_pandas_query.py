from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState

_STUB_FIXED_QUERY = "df.head()  # Phase 2 stub fixed query"


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


def fix_pandas_query_node(state: WorkflowState) -> dict[str, Any]:
    """Agent 2 retry stub — placeholder query fix until Phase 4."""
    planner_output = dict(state.get("planner_output") or {})
    failed_query = planner_output.get("pandas_query")
    execution_result = state.get("execution_result") or {}

    planner_output["pandas_query"] = _STUB_FIXED_QUERY
    planner_output["queries"] = list(planner_output.get("queries") or [])
    planner_output["queries"].append(_STUB_FIXED_QUERY)

    return {
        "status": "executing",
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "fix_pandas_query",
                failed_query=failed_query,
                error=execution_result.get("error"),
            ),
        },
    }
