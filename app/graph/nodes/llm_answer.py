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


def llm_answer_node(state: WorkflowState) -> dict[str, Any]:
    """Document RAG stub — placeholder LLM answer until Phase 5."""
    retrieval_result = dict(state.get("retrieval_result") or {})
    reranked_chunks = list(retrieval_result.get("reranked_chunks") or [])

    retrieval_result["llm_answer"] = (
        "Phase 2 stub document answer based on retrieved context."
    )
    retrieval_result["sources"] = [
        {
            "chunk_id": chunk.get("chunk_id", f"stub-chunk-{index + 1}"),
            "file_path": chunk.get("file_path") or state.get("selected_file_path"),
        }
        for index, chunk in enumerate(reranked_chunks)
    ] or [
        {
            "chunk_id": "stub-chunk-1",
            "file_path": state.get("selected_file_path"),
        }
    ]

    return {
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "llm_answer",
                sources=len(retrieval_result["sources"]),
            ),
        },
    }
