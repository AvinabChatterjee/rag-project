from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.rag.reranker import rerank_chunks


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


def _resolve_retrieval_query(state: WorkflowState) -> str:
    retrieval_result = state.get("retrieval_result") or {}
    planner_output = state.get("planner_output") or {}

    query = (
        retrieval_result.get("query_text")
        or planner_output.get("retrieval_query")
        or state.get("user_question", "")
    )
    normalized = str(query).strip()
    if not normalized:
        raise ValueError(
            "reranker requires retrieval_result.query_text, "
            "planner_output.retrieval_query, or user_question."
        )
    return normalized


def reranker_node(state: WorkflowState) -> dict[str, Any]:
    """Rerank retrieved chunks with a cross-encoder and keep the top-k."""
    retrieval_result = dict(state.get("retrieval_result") or {})
    retrieved_chunks = list(retrieval_result.get("retrieved_chunks") or [])
    retrieval_query = _resolve_retrieval_query(state)

    reranked_chunks = rerank_chunks(retrieval_query, retrieved_chunks)
    retrieval_result["reranked_chunks"] = reranked_chunks

    return {
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "reranker",
                chunks=len(reranked_chunks),
            ),
        },
    }
