from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
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
    planner_output = state.get("planner_output") or {}
    pandas_query = planner_output.get("pandas_query")
    selected_file_path = state.get("selected_file_path")

    if not selected_file_path:
        raise ValueError("code_executor requires selected_file_path from query_planner.")
    if not pandas_query or not str(pandas_query).strip():
        raise ValueError("code_executor requires planner_output.pandas_query.")

    tool_result = run_query_data(selected_file_path, str(pandas_query))
    success = tool_result.get("status") == "success"

    if success:
        execution_result: dict[str, Any] = {
            "success": True,
            "attempts": attempts,
            "raw_result": tool_result.get("result"),
            "executed_code": tool_result.get("executed_code"),
            "error": None,
        }
    else:
        execution_result = {
            "success": False,
            "attempts": attempts,
            "raw_result": None,
            "executed_code": tool_result.get("executed_code"),
            "error": tool_result.get("error", "Pandas execution failed."),
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
                success=success,
                error=execution_result.get("error"),
            ),
        },
    }
