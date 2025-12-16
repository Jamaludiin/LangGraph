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

def static_analyzer_node(state: AgentState):
    response = llm.invoke(
        f"""Analyze the following code and extract:
        \n\n{state['code_under_test']}
        - list of functions
        - expected inputs
        - branches / edge cases
        - possible failures
        - potential edge cases
        - potential edge cases
    )"""

    # Write the static analyzer to a file
    output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/static_analyzer/static_analyzer.md"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(response.content)
    return {"static_analyzer": response.content}