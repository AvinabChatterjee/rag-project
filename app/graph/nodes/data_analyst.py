from datetime import datetime, timezone
from typing import Any, Literal

from app.graph.state import WorkflowState
from app.graph.validation import build_execution_error_message

Confidence = Literal["high", "medium", "low"]


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


def _build_stub_answer(state: WorkflowState) -> tuple[str, str | None, Confidence]:
    route = state.get("route")
    execution_result = state.get("execution_result") or {}
    retrieval_result = state.get("retrieval_result") or {}

    if route == "tabular":
        if execution_result.get("success"):
            return (
                "Phase 2 stub analyst answer from tabular execution result.",
                None,
                "medium",
            )
        planner_output = state.get("planner_output") or {}
        error_message = build_execution_error_message(
            execution_result.get("error") or "Execution failed.",
            planner_output.get("dataset_summary"),
        )
        return (
            "Phase 2 stub analyst message: tabular execution failed.",
            error_message,
            "low",
        )

    if state.get("cache_hit") and retrieval_result.get("cached_answer"):
        return (
            str(retrieval_result["cached_answer"]),
            None,
            "medium",
        )

    llm_answer = retrieval_result.get("llm_answer")
    if llm_answer:
        return (
            str(llm_answer),
            None,
            "medium",
        )

    return (
        "Phase 2 stub analyst answer for document route.",
        None,
        "low",
    )


def data_analyst_node(state: WorkflowState) -> dict[str, Any]:
    """Agent 3 stub — placeholder final answer until Phase 6."""
    final_answer, error_message, confidence = _build_stub_answer(state)

    analyst_output: dict[str, Any] = {
        "final_answer": final_answer,
        "error_message": error_message,
        "confidence": confidence,
    }

    return {
        "status": "completed",
        "analyst_output": analyst_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "data_analyst",
                route=state.get("route"),
                confidence=confidence,
            ),
        },
    }
