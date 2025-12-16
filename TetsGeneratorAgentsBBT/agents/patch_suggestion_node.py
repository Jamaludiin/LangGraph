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
            \n\n{state['execution']}
            \n\nAnalyze the failure
            \n\nPropose code fixes
            \n\nExplain the root cause
        """
    )
    # Write the patch suggestion to a file
    output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/patch_suggestion/patch_suggestion.md"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(response.content)
    return {"patch_suggestion": response.content}