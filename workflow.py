from langgraph.graph import StateGraph, END
from nodes import AgentState, search_products, generate_recommendations, should_continue

def create_workflow() -> StateGraph:
    """Create the LangGraph workflow"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("search_products", search_products)
    workflow.add_node("generate_recommendations", generate_recommendations)
    
    # Add edges
    workflow.add_edge("search_products", "generate_recommendations")
    workflow.add_conditional_edges(
        "generate_recommendations",
        should_continue,
        {
            "continue": END,
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("search_products")
    
    return workflow.compile() 