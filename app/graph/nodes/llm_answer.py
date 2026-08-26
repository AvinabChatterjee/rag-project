from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.llm.openai_client import call_llm
from app.rag.prompts import (
    DOCUMENT_ANSWER_SYSTEM_PROMPT,
    build_document_answer_user_prompt,
)


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


def _build_sources(
    chunks: list[dict[str, Any]],
    *,
    selected_file_path: str | None,
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.get("chunk_id", f"chunk-{index + 1}"),
            "file_path": chunk.get("file_path") or selected_file_path,
        }
        for index, chunk in enumerate(chunks)
    ]


async def llm_answer_node(state: WorkflowState) -> dict[str, Any]:
    """Generate a grounded answer from reranked document chunks."""
    retrieval_result = dict(state.get("retrieval_result") or {})
    reranked_chunks = list(retrieval_result.get("reranked_chunks") or [])

    user_question = (state.get("user_question") or "").strip()
    if not user_question:
        raise ValueError("llm_answer requires user_question.")

    if not reranked_chunks:
        llm_answer = "I don't know."
    else:
        user_prompt = build_document_answer_user_prompt(user_question, reranked_chunks)
        llm_answer = (
            await call_llm(
                DOCUMENT_ANSWER_SYSTEM_PROMPT,
                user_prompt,
                temperature=0.0,
            )
        ).strip()

    retrieval_result["llm_answer"] = llm_answer
    retrieval_result["sources"] = _build_sources(
        reranked_chunks,
        selected_file_path=state.get("selected_file_path"),
    )

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
