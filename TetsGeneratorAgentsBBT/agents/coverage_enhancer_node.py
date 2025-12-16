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
        f"""Read the following source code and test code and check for:
            \n\n{state['code_under_test']}
            \n\n{state['test_plan']}
            \n\n{state['test_case']}
            \n\n{state['static_analyzer']}
            \n\n{state['patch_suggestion']}
            \n\n{state['execution']}
            \n\n{state['mutational_testing']}
            \n\n{state['bug_predictor']}
            \n\nMissing branches
            \n\nMissing error paths
            \n\nUntested input types
            \n\nAdd more tests or request the test generator to improve coverage.
        """
    )
    return {"coverage_enhancer": response.content}