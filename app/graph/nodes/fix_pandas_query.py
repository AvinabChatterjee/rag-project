from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.graph.validation import (
    parse_pandas_response,
    require_failed_execution,
    require_pandas_query,
    require_successful_dataset_summary,
)
from app.llm.openai_client import call_llm_json
from app.rag.prompts import (
    FIX_PANDAS_QUERY_SYSTEM_PROMPT,
    build_fix_pandas_query_user_prompt,
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


async def fix_pandas_query_node(state: WorkflowState) -> dict[str, Any]:
    """Agent 2 retry — LLM fixes a failed Pandas query using the execution error."""
    planner_output = dict(state.get("planner_output") or {})
    execution_result = state.get("execution_result") or {}
    dataset_summary = require_successful_dataset_summary(
        planner_output.get("dataset_summary")
    )

    failed_query = require_pandas_query(planner_output)
    error_message = require_failed_execution(execution_result)

    user_question = state.get("user_question", "").strip()
    if not user_question:
        raise ValueError("fix_pandas_query requires a non-empty user_question.")

    user_prompt = build_fix_pandas_query_user_prompt(
        user_question=user_question,
        dataset_summary=dataset_summary,
        failed_query=failed_query,
        error_message=error_message,
    )
    try:
        llm_response = await call_llm_json(
            FIX_PANDAS_QUERY_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
        )
    except ValueError as exc:
        raise ValueError(f"Pandas fix LLM call failed: {exc}") from exc

    try:
        fixed_query = parse_pandas_response(llm_response, dataset_summary)
    except ValueError as exc:
        raise ValueError(f"Pandas fix response validation failed: {exc}") from exc

    planner_output["pandas_query"] = fixed_query
    planner_output["queries"] = list(planner_output.get("queries") or [])
    planner_output["queries"].append(fixed_query)

    return {
        "status": "executing",
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "fix_pandas_query",
                failed_query=failed_query,
                error=error_message,
                query_preview=fixed_query[:120],
            ),
        },
    }
