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


from agents.AgentState import AgentState
from agents.test_plan_node import test_plan_node
from agents.test_case_generator_node import test_case_generator_node
# ----------------------------------------------------------
# 5️⃣ Build LangGraph workflow
# ----------------------------------------------------------

workflow = StateGraph(AgentState)
workflow.add_node("test_plan_node", test_plan_node)
workflow.add_node("test_case_generator_node", test_case_generator_node)
workflow.add_edge(START, "test_plan_node")
workflow.add_edge("test_plan_node", "test_case_generator_node")
workflow.add_edge("test_case_generator_node", END)
app = workflow.compile()

"""
workflow = StateGraph(AgentState)
workflow.add_node("code_reader_node", code_reader_node)
workflow.add_node("static_analyzer_node", static_analyzer_node)
workflow.add_node("test_plan_node", test_plan_node)
workflow.add_node("test_plan_reviewer_node", test_plan_reviewer_node)
workflow.add_node("test_case_generator_node", test_case_generator_node)
workflow.add_node("test_case_optimizer_node", test_case_optimizer_node)
workflow.add_node("coverage_enhancer_node", coverage_enhancer_node)
workflow.add_node("documentation_generator_node", documentation_generator_node)
workflow.add_node("bug_predictor_node", bug_predictor_node)
workflow.add_node("execution_node", execution_node)
workflow.add_node("patch_suggestion_node", patch_suggestion_node)

workflow.add_edge(START, "code_reader_node")
workflow.add_edge("code_reader_node", "static_analyzer_node")
workflow.add_edge("static_analyzer_node", "test_plan_agent")
workflow.add_edge("test_plan_node", "test_plan_reviewer_node")
workflow.add_edge("test_plan_reviewer_node", "test_case_generator_node")
workflow.add_edge("test_case_generator_node", "test_case_optimizer_node")
workflow.add_edge("test_case_optimizer_node", "coverage_enhancer_node")
workflow.add_edge("coverage_enhancer_node", "documentation_generator_node")
workflow.add_edge("documentation_generator_node", "bug_predictor_node")
workflow.add_edge("bug_predictor_node", "execution_node")
workflow.add_edge("execution_node", "patch_suggestion_node")
workflow.add_edge("patch_suggestion_node", END)
app = workflow.compile()"""