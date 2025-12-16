"""
2️⃣ Test Plan Reviewer Agent

After the *Test Plan Agent* generates a test plan, add a reviewer agent to:
    detect missing edge cases
    validate logic
    ensure completeness
    ensure plan follows testing best practices

Why?
Gives you *two LLM passes* → higher quality output.
"""

from agents.AgentState import AgentState    
from agents.environmentVariables import llm

def test_plan_reviewer_node(state: AgentState):
    response = llm.invoke(
        f"""Review the following test plan and ensure it is complete and follows testing best practices:
            \n\n{state['test_plan']}
            \n\nDetect missing edge cases
            \n\nValidate logic
            \n\nEnsure completeness
            \n\nEnsure plan follows testing best practices
        """
    )
    # Write the test plan reviewer to a file
    output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/test_plan_reviewer/test_plan_reviewer.md"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w") as file:
        file.write(response.content)
    return {"test_plan_reviewer": response.content}
