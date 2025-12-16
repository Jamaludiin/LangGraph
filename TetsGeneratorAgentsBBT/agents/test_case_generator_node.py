


from agents.AgentState import AgentState
from agents.environmentVariables import llm

# accepts the test plan and generates the test case
def test_case_generator_node(state: AgentState):
    code_under_test = state["code_under_test"]
    response = llm.invoke(
        f"""Generate only the pure python code for pytest style test cases, 
        including necessary imports and comments, for the following code and plan:\n\n{code_under_test}\n\n{state['test_plan']}
        please return only the pure python code, no other text or comments.
        no additional prefix or postfix or describtions
        do not include the code these ``` or python keywords
        only procide the code only no additional prefix or postfix or describtions
        do not include the code these ``` or python keywords
        only procide the code only no additional prefix or postfix or describtions
        do not include the code these ``` or python keywords"""
    )
    return {"test_case": response.content}