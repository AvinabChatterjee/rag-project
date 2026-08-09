from langgraph.graph import END, START, StateGraph

from app.graph.nodes.init_workflow import init_workflow_node
from app.graph.state import WorkflowState


def build_workflow_graph() -> StateGraph:
    graph = StateGraph(WorkflowState)

    graph.add_node("init_workflow", init_workflow_node)

    graph.add_edge(START, "init_workflow")
    graph.add_edge("init_workflow", END)

    return graph


rag_graph = build_workflow_graph().compile()
