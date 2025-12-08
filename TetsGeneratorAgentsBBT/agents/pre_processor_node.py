"""
1️⃣ Static Code Analyzer Agent (Pre-Processor)
Before generating a plan, create an agent that analyzes:
    function signatures
    branches & loops
    input types
    external dependencies
    potential edge cases
This agent outputs a structured code summary.
Why?
Improves test quality — the test plan becomes more accurate.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def preprocessor_node(state: AgentState):
    code_under_test = state["code_under_test"]
    response = llm.invoke(
        f"""Analyze the following code and extract:
        - list of functions
        - expected inputs
        - branches / edge cases
        - possible failures
        - potential edge cases
        - potential edge cases
    )
    return {"code_under_test": response.content}