"""
3️⃣ Test Case Optimizer Agent

After test case generation, add another agent that:
    refactors the test code
    ensures `pytest` naming conventions
    removes unnecessary mocks
    checks import statements
    ensures readability and maintainability
"""

from agents.AgentState import AgentState
from agents.environmentVariables import llm

def test_case_optimizer_node(state: AgentState): 
    response = llm.invoke(
        f"""Refactor the following test code to ensure it is complete and follows testing best practices:
            \n\n{state['test_case']}
            \n\nEnsure `pytest` naming conventions
            \n\nRemove unnecessary mocks
            \n\nCheck import statements
            \n\nEnsure readability and maintainability
            \n\nReturn only the pure python code, no other text or comments.
            \n\nno additional prefix or postfix or describtions
            \n\ndo not include the code these ``` or python keywords
            \n\nonly procide the code only no additional prefix or postfix or describtions
            \n\ndo not include the code these ``` or python keywords

        """
        # Write the test case optimizer to a file
        output_file_path = "/Users/LLM and HuggingFace/LangGraph/TetsGeneratorAgentsBBT/AgentFeedback/test_case_optimizer/test_case_optimizer.md"
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, "w") as file:
            file.write(response.content)
    )
    return {"test_case_optimizer": response.content}