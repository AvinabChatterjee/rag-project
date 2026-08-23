from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph.state import WorkflowState
from app.llm.openai_client import embed_text
from app.rag.cache import lookup_semantic_cache


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


async def cache_lookup_node(state: WorkflowState) -> dict[str, Any]:
    """Embed the retrieval query and check the semantic cache for this file."""
    planner_output = state.get("planner_output") or {}
    retrieval_query = (planner_output.get("retrieval_query") or "").strip()
    if not retrieval_query:
        raise ValueError("cache_lookup requires planner_output.retrieval_query.")

    selected_file_path = state.get("selected_file_path")
    if not selected_file_path:
        raise ValueError("cache_lookup requires selected_file_path.")

    resolved_path = str(Path(selected_file_path).resolve())
    query_embedding = await embed_text(retrieval_query)
    cache_hit, cached_answer, best_score = lookup_semantic_cache(
        resolved_path,
        query_embedding,
    )

    retrieval_result = _base_retrieval_result(state)
    retrieval_result.update(
        {
            "cache_hit": cache_hit,
            "cached_answer": cached_answer if cache_hit else None,
            "query_text": retrieval_query,
            "query_embedding": query_embedding,
            "retrieved_chunks": retrieval_result.get("retrieved_chunks", []),
            "reranked_chunks": retrieval_result.get("reranked_chunks", []),
            "llm_answer": retrieval_result.get("llm_answer"),
            "sources": retrieval_result.get("sources", []),
        }
    )

    return {
        "status": "retrieving",
        "cache_hit": cache_hit,
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "cache_lookup",
                cache_hit=cache_hit,
                similarity_score=best_score,
            ),
        },
    }
