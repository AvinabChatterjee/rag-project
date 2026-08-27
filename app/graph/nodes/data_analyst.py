from datetime import datetime, timezone
from typing import Any, Literal

from app.graph.state import WorkflowState
from app.graph.validation import build_execution_error_message, parse_analyst_response
from app.llm.openai_client import call_llm_json
from app.rag.prompts import DATA_ANALYST_SYSTEM_PROMPT, build_data_analyst_user_prompt

Confidence = Literal["high", "medium", "low"]
_ANALYST_TEMPERATURE = 0.3


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


def _build_analyst_prompt(
    state: WorkflowState,
) -> tuple[str, str | None]:
    user_question = (state.get("user_question") or "").strip()
    route = state.get("route")
    execution_result = state.get("execution_result") or {}
    retrieval_result = state.get("retrieval_result") or {}
    planner_output = state.get("planner_output") or {}
    dataset_summary = planner_output.get("dataset_summary")

    if route == "tabular":
        if execution_result.get("success"):
            return (
                build_data_analyst_user_prompt(
                    user_question,
                    raw_result=execution_result.get("raw_result"),
                ),
                None,
            )

        error_message = build_execution_error_message(
            str(execution_result.get("error") or "Execution failed."),
            dataset_summary,
        )
        return (
            build_data_analyst_user_prompt(
                user_question,
                execution_error=error_message,
                dataset_summary=dataset_summary,
            ),
            error_message,
        )

    if state.get("cache_hit") and retrieval_result.get("cached_answer"):
        return (
            build_data_analyst_user_prompt(
                user_question,
                cached_answer=str(retrieval_result["cached_answer"]),
            ),
            None,
        )

    if retrieval_result.get("llm_answer"):
        return (
            build_data_analyst_user_prompt(
                user_question,
                llm_answer=str(retrieval_result["llm_answer"]),
                sources=list(retrieval_result.get("sources") or []),
            ),
            None,
        )

    return build_data_analyst_user_prompt(user_question), None


async def data_analyst_node(state: WorkflowState) -> dict[str, Any]:
    """Agent 3 — convert raw workflow results into a human-friendly answer."""
    user_question = (state.get("user_question") or "").strip()
    if not user_question:
        raise ValueError("data_analyst requires user_question.")

    user_prompt, preset_error_message = _build_analyst_prompt(state)
    llm_response = await call_llm_json(
        DATA_ANALYST_SYSTEM_PROMPT,
        user_prompt,
        temperature=_ANALYST_TEMPERATURE,
    )
    analyst_output = parse_analyst_response(llm_response)

    if preset_error_message is not None:
        analyst_output["error_message"] = preset_error_message

    confidence = analyst_output["confidence"]

    return {
        "status": "completed",
        "analyst_output": analyst_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "data_analyst",
                route=state.get("route"),
                confidence=confidence,
            ),
        },
    }
