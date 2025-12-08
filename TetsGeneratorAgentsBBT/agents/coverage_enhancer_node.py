"""
5️⃣ Coverage Enhancer Agent
After test case generation, add a coverage enhancer agent that reads the generated test code and checks for:
    missing branches
    missing error paths
    untested input types

It adds more tests or requests the test generator to improve coverage.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def coverage_enhancer_node(state: AgentState): 
    response = llm.invoke(
        f"""Read the following test code and check for:
            \n\n{state['test_case']}
            \n\nMissing branches
            \n\nMissing error paths
            \n\nUntested input types
            \n\nAdd more tests or request the test generator to improve coverage.
        """
    )
    return {"coverage_enhancer": response.content}