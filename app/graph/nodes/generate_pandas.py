from datetime import datetime, timezone
from typing import Any

from app.graph.state import WorkflowState
from app.llm.openai_client import call_llm_json
from app.rag.prompts import (
    PANDAS_GENERATOR_SYSTEM_PROMPT,
    build_pandas_generator_user_prompt,
)

_FORBIDDEN_PATTERNS = ("read_csv", "read_excel", "open(")


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


def _parse_pandas_query(response: dict[str, Any]) -> str:
    pandas_query = response.get("pandas_query")
    if not isinstance(pandas_query, str) or not pandas_query.strip():
        raise ValueError("LLM response missing non-empty 'pandas_query'.")

    query = pandas_query.strip()
    lowered = query.lower()
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in lowered:
            raise ValueError(
                f"Generated Pandas query must not contain '{pattern}'."
            )
    return query


async def generate_pandas_node(state: WorkflowState) -> dict[str, Any]:
    """Tabular path — generate Pandas code from the inspected dataset summary."""
    planner_output = dict(state.get("planner_output") or {})
    dataset_summary = planner_output.get("dataset_summary")

    if not dataset_summary or dataset_summary.get("status") != "success":
        error = (dataset_summary or {}).get("error", "Dataset summary missing or failed.")
        raise ValueError(f"Cannot generate Pandas query: {error}")

    user_question = state["user_question"]
    user_prompt = build_pandas_generator_user_prompt(user_question, dataset_summary)
    llm_response = await call_llm_json(
        PANDAS_GENERATOR_SYSTEM_PROMPT,
        user_prompt,
        temperature=0.0,
    )
    pandas_query = _parse_pandas_query(llm_response)

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
