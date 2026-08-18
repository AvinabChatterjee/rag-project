from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.graph.validation import parse_pandas_response, require_successful_dataset_summary
from app.llm.openai_client import call_llm_json
from app.rag.prompts import (
    PANDAS_GENERATOR_SYSTEM_PROMPT,
    build_pandas_generator_user_prompt,
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


async def generate_pandas_node(state: WorkflowState) -> dict[str, Any]:
    """Tabular path — generate Pandas code from the inspected dataset summary."""
    planner_output = dict(state.get("planner_output") or {})
    dataset_summary = require_successful_dataset_summary(
        planner_output.get("dataset_summary")
    )

    user_question = state.get("user_question", "").strip()
    if not user_question:
        raise ValueError("generate_pandas requires a non-empty user_question.")

    user_prompt = build_pandas_generator_user_prompt(user_question, dataset_summary)
    try:
        llm_response = await call_llm_json(
            PANDAS_GENERATOR_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
        )
    except ValueError as exc:
        raise ValueError(f"Pandas generator LLM call failed: {exc}") from exc

    pandas_query = parse_pandas_response(llm_response, dataset_summary)

    planner_output["pandas_query"] = pandas_query
    planner_output["queries"] = list(planner_output.get("queries") or [])
    planner_output["queries"].append(pandas_query)

    return {
        "status": "executing",
        "planner_output": planner_output,
        "metadata": {
            **(state.get("metadata") or {}),
            "agent_trace": _append_trace(
                state,
                "generate_pandas",
                query_preview=pandas_query[:120],
            ),
        },
    }
