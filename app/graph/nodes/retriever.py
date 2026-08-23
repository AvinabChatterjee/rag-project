from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph.state import WorkflowState
from app.llm.openai_client import embed_text
from app.rag.retriever import retrieve_chunks


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


async def retriever_node(state: WorkflowState) -> dict[str, Any]:
    """Query Chroma for top-k chunks using the cached query embedding."""
    retrieval_result = dict(state.get("retrieval_result") or {})
    query_embedding = retrieval_result.get("query_embedding")

    if not query_embedding:
        planner_output = state.get("planner_output") or {}
        retrieval_query = (planner_output.get("retrieval_query") or "").strip()
        if not retrieval_query:
            raise ValueError(
                "retriever requires retrieval_result.query_embedding or "
                "planner_output.retrieval_query."
            )
        query_embedding = await embed_text(retrieval_query)
        retrieval_result["query_embedding"] = query_embedding

    selected_file_path = state.get("selected_file_path")
    if not selected_file_path:
        raise ValueError("retriever requires selected_file_path.")

    resolved_path = str(Path(selected_file_path).resolve())
    retrieved_chunks = retrieve_chunks(resolved_path, query_embedding)
    retrieval_result["retrieved_chunks"] = retrieved_chunks

    return {
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "retriever",
                chunks=len(retrieved_chunks),
            ),
        },
    }
