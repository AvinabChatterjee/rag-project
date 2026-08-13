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


def _base_retrieval_result(state: WorkflowState) -> dict[str, Any]:
    return dict(state.get("retrieval_result") or {})


def cache_lookup_node(state: WorkflowState) -> dict[str, Any]:
    """Document RAG stub — always cache miss until Phase 5."""
    retrieval_result = _base_retrieval_result(state)
    retrieval_result.update(
        {
            "cache_hit": False,
            "cached_answer": None,
            "retrieved_chunks": retrieval_result.get("retrieved_chunks", []),
            "reranked_chunks": retrieval_result.get("reranked_chunks", []),
            "llm_answer": retrieval_result.get("llm_answer"),
            "sources": retrieval_result.get("sources", []),
        }
    )

    return {
        "status": "retrieving",
        "cache_hit": False,
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(state, "cache_lookup", cache_hit=False),
        },
    }
