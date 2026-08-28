from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.graph.validation import (
    AnalystScenario,
    build_execution_error_message,
    determine_analyst_scenario,
    finalize_analyst_output,
    parse_analyst_response,
)
from app.llm.openai_client import call_llm_json
from app.rag.prompts import DATA_ANALYST_SYSTEM_PROMPT, build_data_analyst_user_prompt

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
    scenario: AnalystScenario,
) -> tuple[str, str | None, list[dict[str, Any]]]:
    user_question = (state.get("user_question") or "").strip()
    execution_result = state.get("execution_result") or {}
    retrieval_result = state.get("retrieval_result") or {}
    planner_output = state.get("planner_output") or {}
    dataset_summary = planner_output.get("dataset_summary")
    sources = list(retrieval_result.get("sources") or [])

    if scenario == "tabular_success":
        return (
            build_data_analyst_user_prompt(
                user_question,
                scenario=scenario,
                raw_result=execution_result.get("raw_result"),
            ),
            None,
            sources,
        )

    if scenario == "tabular_failure":
        error_message = build_execution_error_message(
            str(execution_result.get("error") or "Execution failed."),
            dataset_summary,
        )
        return (
            build_data_analyst_user_prompt(
                user_question,
                scenario=scenario,
                execution_error=error_message,
                dataset_summary=dataset_summary,
            ),
            error_message,
            sources,
        )

    if scenario == "document_cache_hit":
        return (
            build_data_analyst_user_prompt(
                user_question,
                scenario=scenario,
                cached_answer=str(retrieval_result["cached_answer"]),
            ),
            None,
            sources,
        )

    if scenario == "document_cache_miss":
        return (
            build_data_analyst_user_prompt(
                user_question,
                scenario=scenario,
                llm_answer=str(retrieval_result["llm_answer"]),
                sources=sources,
            ),
            None,
            sources,
        )

    return (
        build_data_analyst_user_prompt(user_question, scenario=scenario),
        None,
        sources,
    )


async def data_analyst_node(state: WorkflowState) -> dict[str, Any]:
    """Agent 3 — convert raw workflow results into a human-friendly answer."""
    user_question = (state.get("user_question") or "").strip()
    if not user_question:
        raise ValueError("data_analyst requires user_question.")

    scenario = determine_analyst_scenario(state)
    user_prompt, preset_error_message, sources = _build_analyst_prompt(state, scenario)
    llm_response = await call_llm_json(
        DATA_ANALYST_SYSTEM_PROMPT,
        user_prompt,
        temperature=_ANALYST_TEMPERATURE,
    )
    analyst_output = finalize_analyst_output(
        parse_analyst_response(llm_response),
        scenario=scenario,
        sources=sources,
        preset_error_message=preset_error_message,
    )

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
                scenario=scenario,
                confidence=confidence,
            ),
        },
    }
