from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.graph.validation import parse_planner_response
from app.llm.openai_client import call_llm_json
from app.rag.prompts import (
    QUERY_PLANNER_SYSTEM_PROMPT,
    build_query_planner_user_prompt,
)

_TABULAR_TYPES = frozenset({"csv", "excel"})


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


async def query_planner_node(state: WorkflowState) -> dict[str, Any]:
    """
    Agent 1 — LLM decides route, picks a file, and sets retrieval_query.
    """
    available_files = state.get("available_files") or []
    if not available_files:
        raise ValueError("query_planner requires available_files from init_workflow.")

    user_question = state.get("user_question", "").strip()
    if not user_question:
        raise ValueError("query_planner requires a non-empty user_question.")

    user_prompt = build_query_planner_user_prompt(user_question, available_files)
    try:
        llm_response = await call_llm_json(
            QUERY_PLANNER_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
        )
    except ValueError as exc:
        raise ValueError(f"Query planner LLM call failed: {exc}") from exc

    route, selected_file, reasoning, retrieval_query = parse_planner_response(
        llm_response,
        available_files,
        user_question,
    )
    file_type = selected_file["file_type"]
    if file_type not in (*_TABULAR_TYPES, "document"):
        raise ValueError(f"Unsupported file type for routing: {file_type!r}")

    planner_output: dict[str, Any] = {
        "route": route,
        "reasoning": reasoning,
        "dataset_summary": None,
        "pandas_query": None,
        "retrieval_query": retrieval_query,
        "queries": [],
    }

    return {
        "status": "planning",
        "route": route,
        "selected_file_path": selected_file["file_path"],
        "selected_file_type": file_type,  # type: ignore[typeddict-item]
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "query_planner",
                route=route,
                selected_file=selected_file["file_name"],
            ),
        },
    }
