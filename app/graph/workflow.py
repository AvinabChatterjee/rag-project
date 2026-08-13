from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    cache_lookup_node,
    cache_store_node,
    code_executor_node,
    data_analyst_node,
    fix_pandas_query_node,
    generate_pandas_node,
    init_workflow_node,
    inspect_dataset_node,
    llm_answer_node,
    query_planner_node,
    reranker_node,
    retriever_node,
)
from app.graph.routing import (
    route_after_cache,
    route_after_execution,
    route_after_planner,
)
from app.graph.state import WorkflowState


def build_workflow_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("init_workflow", init_workflow_node)
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("inspect_dataset", inspect_dataset_node)
    graph.add_node("generate_pandas", generate_pandas_node)
    graph.add_node("code_executor", code_executor_node)
    graph.add_node("fix_pandas_query", fix_pandas_query_node)
    graph.add_node("cache_lookup", cache_lookup_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("llm_answer", llm_answer_node)
    graph.add_node("cache_store", cache_store_node)
    graph.add_node("data_analyst", data_analyst_node)

    graph.add_edge(START, "init_workflow")
    graph.add_edge("init_workflow", "query_planner")

    graph.add_conditional_edges(
        "query_planner",
        route_after_planner,
        {
            "tabular": "inspect_dataset",
            "document": "cache_lookup",
        },
    )

    graph.add_edge("inspect_dataset", "generate_pandas")
    graph.add_edge("generate_pandas", "code_executor")
    graph.add_conditional_edges(
        "code_executor",
        route_after_execution,
        {
            "retry": "fix_pandas_query",
            "analyze": "data_analyst",
        },
    )
    graph.add_edge("fix_pandas_query", "code_executor")

    graph.add_conditional_edges(
        "cache_lookup",
        route_after_cache,
        {
            "hit": "data_analyst",
            "miss": "retriever",
        },
    )
    graph.add_edge("retriever", "reranker")
    graph.add_edge("reranker", "llm_answer")
    graph.add_edge("llm_answer", "cache_store")
    graph.add_edge("cache_store", "data_analyst")
    graph.add_edge("data_analyst", END)

    return graph


rag_graph = build_workflow_graph().compile()
