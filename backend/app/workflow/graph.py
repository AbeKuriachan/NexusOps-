from langgraph.graph import StateGraph, START, END
from app.workflow.state import RAGState
from app.workflow.nodes import (
    classify_query,
    retrieve_vector,
    retrieve_graph,
    build_context,
    refuse_query,
    generate_answer
)

def route_query(state: RAGState):
    """
    Router based on the query classification type.
    """
    q_type = state.get("query_type")
    if q_type == "VECTOR_ONLY":
        return "vector"
    elif q_type == "GRAPH_ONLY":
        return "graph"
    elif q_type == "HYBRID":
        # Launch vector and graph retrieval in parallel
        return ["vector", "graph"]
    else:
        return "refusal"

# Construct State Graph
workflow = StateGraph(RAGState)

# Add Nodes
workflow.add_node("classify", classify_query)
workflow.add_node("vector", retrieve_vector)
workflow.add_node("graph", retrieve_graph)
workflow.add_node("context", build_context)
workflow.add_node("refusal", refuse_query)
workflow.add_node("answer", generate_answer)

# Set Entry Point
workflow.add_edge(START, "classify")

# Add Routing Edges
workflow.add_conditional_edges(
    "classify",
    route_query,
    {
        "vector": "vector",
        "graph": "graph",
        "refusal": "refusal"
    }
)

# Connect retrievers to context builder
workflow.add_edge("vector", "context")
workflow.add_edge("graph", "context")

# Connect context builder to answer generator
workflow.add_edge("context", "answer")

# Connect end points
workflow.add_edge("answer", END)
workflow.add_edge("refusal", END)

# Compile Graph
app_workflow = workflow.compile()
