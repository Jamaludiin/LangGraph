"""
8️⃣ Execution Agent (Optional)
If you allow running code:
    actually executes tests
    returns failures
    triggers re-generation

(*Use carefully! Very powerful step.*)
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def execution_node(state: AgentState):
    response = llm.invoke(
        f"""Execute the following test code and return the failures:
            \n\n{state['test_case']}
        """
    )
    return {"execution": response.content}