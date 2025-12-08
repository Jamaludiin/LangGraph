"""
6️⃣ Documentation Generator Agent
After test case generation, add a documentation generator agent that generates:
    test documentation
    test-report.md
    explanation of the logic
    how tests map to functions

Perfect for DevOps workflows.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def documentation_generator_node(state: AgentState):
    response = llm.invoke(
        f"""Generate test documentation, test-report.md, explanation of the logic, and how tests map to functions:
            \n\n{state['test_case']}
        """
    )
    return {"documentation_generator": response.content}