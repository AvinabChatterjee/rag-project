from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph.state import WorkflowState
from app.rag.cache import insert_cache_entry, purge_expired_cache_entries

_DONT_CACHE_ANSWERS = frozenset({"i don't know.", "i don't know"})


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


def _resolve_query_text(state: WorkflowState, retrieval_result: dict[str, Any]) -> str:
    planner_output = state.get("planner_output") or {}
    query = (
        retrieval_result.get("query_text")
        or planner_output.get("retrieval_query")
        or ""
    )
    return str(query).strip()


def _should_store_cache_entry(
    state: WorkflowState,
    retrieval_result: dict[str, Any],
) -> tuple[bool, str | None]:
    if state.get("cache_hit"):
        return False, "cache_hit"

    selected_file_path = state.get("selected_file_path")
    if not selected_file_path:
        return False, "missing_file_path"

    query_text = _resolve_query_text(state, retrieval_result)
    if not query_text:
        return False, "missing_query_text"

    query_embedding = retrieval_result.get("query_embedding")
    if not query_embedding:
        return False, "missing_query_embedding"

    llm_answer = str(retrieval_result.get("llm_answer") or "").strip()
    if not llm_answer:
        return False, "missing_llm_answer"
    if llm_answer.lower() in _DONT_CACHE_ANSWERS:
        return False, "non_cacheable_answer"

    return True, None


def cache_store_node(state: WorkflowState) -> dict[str, Any]:
    """Persist the LLM answer in the semantic cache after a cache miss."""
    retrieval_result = dict(state.get("retrieval_result") or {})
    should_store, skip_reason = _should_store_cache_entry(state, retrieval_result)

    stored = False
    cache_entry_id: str | None = None
    if should_store:
        purge_expired_cache_entries()
        cache_entry_id = insert_cache_entry(
            str(Path(state["selected_file_path"]).resolve()),
            _resolve_query_text(state, retrieval_result),
            retrieval_result["query_embedding"],
            str(retrieval_result["llm_answer"]).strip(),
        )
        stored = True

    return {
        "retrieval_result": retrieval_result,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "cache_store",
                stored=stored,
                cache_entry_id=cache_entry_id,
                skip_reason=skip_reason,
                query=_resolve_query_text(state, retrieval_result) or None,
            ),
        },
    }
