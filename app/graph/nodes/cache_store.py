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


def cache_store_node(state: WorkflowState) -> dict[str, Any]:
    """Document RAG stub — no-op cache write until Phase 5."""
    planner_output = state.get("planner_output") or {}
    retrieval_result = dict(state.get("retrieval_result") or {})

    return {
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "cache_store",
                stored=False,
                query=planner_output.get("retrieval_query"),
            ),
        },
    }
