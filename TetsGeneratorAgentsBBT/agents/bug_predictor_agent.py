"""
7️⃣ Bug Predictor Agent
This agent inspects the code and predicts potential defects using static heuristics.

Example:
    division without zero handling
    missing input validation
    unsafe type conversions

Then it asks the test plan generator to create tests around predicted bugs.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def bug_predictor_agent(state: AgentState):
    response = llm.invoke(
        f"""Inspect the following code and predict potential defects using static heuristics:
            \n\n{state['code_under_test']}
            \n\nDivision without zero handling
            \n\nMissing input validation
            \n\nUnsafe type conversions
            \n\nAsk the test plan generator to create tests around predicted bugs.
        """
    )
    return {"bug_predictor": response.content}