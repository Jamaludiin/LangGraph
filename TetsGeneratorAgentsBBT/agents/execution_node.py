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
    # Write the execution to a file
    output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/execution/execution.md"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(response.content)
    return {"execution": response.content}