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


def code_executor_node(state: WorkflowState) -> dict[str, Any]:
    """
    Agent 2 stub — simulates a successful Pandas run until Phase 4.

    Increments execution_attempts and writes a placeholder execution_result.
    """
    attempts = state.get("execution_attempts", 0) + 1
    planner_output = state.get("planner_output") or {}
    executed_code = planner_output.get("pandas_query") or "df.head()  # Phase 2 stub query"

    execution_result: dict[str, Any] = {
        "success": True,
        "attempts": attempts,
        "raw_result": [{"message": "Phase 2 stub execution result"}],
        "executed_code": executed_code,
        "error": None,
    }

    return {
        "status": "executing",
        "execution_attempts": attempts,
        "execution_result": execution_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "code_executor",
                attempts=attempts,
                success=True,
            ),
        },
    }
