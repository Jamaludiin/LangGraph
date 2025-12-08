import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)

from langgraph.graph import StateGraph, START, END

from agents.AgentState import AgentState
from agents.test_plan_node import test_plan_node
from agents.test_case_node import test_case_node
# ----------------------------------------------------------
# 5️⃣ Build LangGraph workflow
# ----------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("test_plan_node", test_plan_node)
workflow.add_node("test_case_node", test_case_node)
workflow.add_edge(START, "test_plan_node")
workflow.add_edge("test_plan_node", "test_case_node")
workflow.add_edge("test_case_node", END)
app = workflow.compile()