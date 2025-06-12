
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from customer_nodes import (
    customer_profile_node,
    summarize_profile_node,
    recommendation_node
)

class WorkflowState(TypedDict):
    query: str
    documents: List[str]
    summary: str
    recommendations: List[str]

builder = StateGraph(WorkflowState)
builder.add_node("profile", customer_profile_node)
builder.add_node("summarize", summarize_profile_node)
builder.add_node("recommend", recommendation_node)

builder.set_entry_point("profile")
builder.add_edge("profile", "summarize")
builder.add_edge("summarize", "recommend")
builder.add_edge("recommend", END)

app = builder.compile()

if __name__ == "__main__":
    result = app.invoke({"query": "preferences of customer Jane Smith"})
    print(result)
