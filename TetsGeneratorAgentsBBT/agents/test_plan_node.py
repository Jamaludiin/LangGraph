


from agents.AgentState import AgentState
from agents.environmentVariables import llm

def test_plan_node(state: AgentState):
    code_under_test = state["code_under_test"]
    response = llm.invoke(
        f"Generate a test plan (positive and negative test cases) for the following code:\n\n{code_under_test}"
    )
    # Write the test plan to a file
    with open("/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/test_plan/test_plan.md", "w") as file:
        file.write(response.content)
    return {"test_plan": response.content}
    
    
