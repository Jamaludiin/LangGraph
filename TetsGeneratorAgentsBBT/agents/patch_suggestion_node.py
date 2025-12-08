"""
9️⃣ Patch Suggestion Agent
If tests fail:
    analyze failure
    propose code fixes
    explain root cause

This turns your project into a **mini autonomous debugging system**.
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def patch_suggestion_node(state: AgentState):
    response = llm.invoke(
        f"""Analyze the following failure and propose code fixes and explain the root cause:
            \n\n{state['failure']}
            \n\nAnalyze the failure
            \n\nPropose code fixes
            \n\nExplain the root cause
        """
    )
    return {"patch_suggestion": response.content}