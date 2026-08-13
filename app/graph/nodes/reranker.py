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


def reranker_node(state: WorkflowState) -> dict[str, Any]:
    """Document RAG stub — placeholder reranked chunks until Phase 5."""
    retrieval_result = dict(state.get("retrieval_result") or {})
    retrieved_chunks = list(retrieval_result.get("retrieved_chunks") or [])
    retrieval_result["reranked_chunks"] = retrieved_chunks[:1] or [
        {
            "chunk_id": "stub-chunk-1",
            "text": "Phase 2 stub reranked chunk.",
            "score": 0.95,
            "file_path": state.get("selected_file_path"),
        }
    ]

    return {
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "reranker",
                chunks=len(retrieval_result["reranked_chunks"]),
            ),
        },
    }
