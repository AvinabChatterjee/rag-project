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


def retriever_node(state: WorkflowState) -> dict[str, Any]:
    """Document RAG stub — placeholder retrieved chunks until Phase 5."""
    retrieval_result = dict(state.get("retrieval_result") or {})
    retrieval_result["retrieved_chunks"] = [
        {
            "chunk_id": "stub-chunk-1",
            "text": "Phase 2 stub retrieved chunk.",
            "score": 0.9,
            "file_path": state.get("selected_file_path"),
        }
    ]

    return {
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "retriever",
                chunks=len(retrieval_result["retrieved_chunks"]),
            ),
        },
    }
