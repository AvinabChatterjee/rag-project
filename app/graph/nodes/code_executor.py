from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.graph.validation import require_pandas_query, require_selected_file_path
from app.tools.dataframe_tools import run_query_data


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
    """Agent 2 — execute generated Pandas code on the selected tabular file."""
    attempts = state.get("execution_attempts", 0) + 1
    max_attempts = state.get("max_execution_attempts", 2)
    planner_output = state.get("planner_output") or {}

    selected_file_path = require_selected_file_path(state.get("selected_file_path"))
    pandas_query = require_pandas_query(planner_output)

    tool_result = run_query_data(selected_file_path, pandas_query)
    success = tool_result.get("status") == "success"

    if success:
        execution_result: dict[str, Any] = {
            "success": True,
            "attempts": attempts,
            "raw_result": tool_result.get("result"),
            "executed_code": tool_result.get("executed_code"),
            "error": None,
        }
        status = "executing"
    else:
        execution_result = {
            "success": False,
            "attempts": attempts,
            "raw_result": None,
            "executed_code": tool_result.get("executed_code"),
            "error": tool_result.get("error", "Pandas execution failed."),
        }
        status = "failed" if attempts >= max_attempts else "executing"

    return {
        "status": status,
        "execution_attempts": attempts,
        "execution_result": execution_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "code_executor",
                attempts=attempts,
                success=success,
                error=execution_result.get("error"),
            ),
        },
    }
