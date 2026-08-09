"""Conditional edge routers for the LangGraph workflow."""

from typing import Literal

from app.graph.state import WorkflowState

PlannerRoute = Literal["tabular", "document"]
ExecutionRoute = Literal["retry", "analyze"]
CacheRoute = Literal["hit", "miss"]


def route_after_planner(state: WorkflowState) -> PlannerRoute:
    """Route tabular vs document after Agent 1 (Query Planner)."""
    planner_output = state.get("planner_output") or {}
    route = planner_output.get("route") or state.get("route")

    if route not in ("tabular", "document"):
        raise ValueError(
            f"Invalid or missing planner route: {route!r}. "
            'Expected "tabular" or "document".'
        )
    return route


def route_after_execution(state: WorkflowState) -> ExecutionRoute:
    """
    After code executor: retry once on failure, otherwise go to analyst.

    Success → analyze.
    Failure and attempts < max → retry (fix_pandas_query).
    Otherwise → analyze (pass error to analyst; never invent data).
    """
    execution_result = state.get("execution_result") or {}
    if execution_result.get("success"):
        return "analyze"

    attempts = state.get("execution_attempts", 0)
    max_attempts = state.get("max_execution_attempts", 2)
    if attempts < max_attempts:
        return "retry"
    return "analyze"


def route_after_cache(state: WorkflowState) -> CacheRoute:
    """After semantic cache lookup: hit skips retrieval; miss continues RAG."""
    return "hit" if state.get("cache_hit") else "miss"
